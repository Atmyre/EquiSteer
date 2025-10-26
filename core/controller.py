import logging
import warnings
import numpy as np
import torch
import abc
import typing as tp
from collections import defaultdict
from typing import Optional, Any

import enum
from core.math import convert_to_widest_dtype
import pickle
import os

logger = logging.getLogger()

EPS = 1e-6

class DiffusionVectorControlMode(enum.StrEnum):
    ATTN_OUTPUT = 'attn_output'
    ATTN_HEADS = 'attn_head'


class ModelToSteer(enum.StrEnum):
    UNET = 'unet'
    LLAMA = 'llama'


class VectorControl(abc.ABC):
    def __init__(self, mode: DiffusionVectorControlMode = None, num_layers: int = None, attribute=None):
        self._mode = mode
        self._active = True
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        self.num_attn_layers = num_layers
        self.attribute = attribute
        if attribute == 'gender':
            self.coin = torch.rand(1) < 0.5
        elif attribute == 'race':
            torch.randint(0, 5, (1,)).item()

        self.debias = False

    @property
    def active(self) -> bool:
        return self._active
    
    @active.setter
    def active(self, value: bool):
        self._active = value
    
    def reset(self):
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        if self.attribute == 'gender':
            self.coin = torch.rand(1) < 0.5
        elif self.attribute == 'race':
            torch.randint(0, 5, (1,)).item()
        self.debias = False
    
    @abc.abstractmethod
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int, min_token_index: int = None):
        raise NotImplementedError

    def __call__(self, vector: torch.Tensor, place_in_unet: str):
        if not self.active:
            return vector
            
        block_index = self._current_position[place_in_unet]
        input_shape = vector.shape
        vector = self.forward(vector, self._diffusion_step, place_in_unet, block_index)
        assert vector.shape == input_shape
        self._current_position[place_in_unet] += 1

        self._current_attn_layer += 1
        if self._current_attn_layer == self.num_attn_layers:
            self._current_attn_layer = 0
            self._current_position = defaultdict(int)
            self._diffusion_step += 1
        return vector

# For each diffusion step,
# for each place in the network represented as string key,
# for each layer position, we store steering vector
SteeringVectors = tp.NewType('SteeringVectors', dict[int, dict[str, list[torch.Tensor]]])

class CrossAttentionOutputSteering(VectorControl):
    def __init__(
        self,
        model_to_steer: ModelToSteer,
        *,
        source_concepts: list[SteeringVectors],
        target_concepts: list[SteeringVectors | None],
        strength: float,

        mode: DiffusionVectorControlMode = None,
        steer_type: str = None,

        steer_only_up=False, 
        steer_back: bool = False,
        device: Any,
        num_layers: int = None,
        renormalize_after_steering: bool = False,
        use_first_diffusion_step: bool = False,
        save_vectors: bool = False,
        save_vectors_path: str = None,
        attribute: str = None,
        model_name: str = None,
        llm: bool = False, # for llm based decision for debiasing -- independent of threshold
        debias_type: str = 'discrete',
    ):
        super().__init__(mode=mode, num_layers=num_layers)
        self.device = device
        
        self.steer_only_up = steer_only_up
        self.steer_back = steer_back
        self.steer_type = steer_type
        self.renormalize_after_steering = renormalize_after_steering
        self.strength = strength
        self.use_first_diffusion_step = use_first_diffusion_step
        self.save_vectors = save_vectors
        self.save_vectors_path = save_vectors_path
        self.model_name = model_name
        self.attribute = attribute
        self.llm = llm
        self.counter = 0 # for saving in forward call
        self.debias = False
        self.debias_type = debias_type

        if self.attribute == 'gender':
            self.coin = torch.rand(1) < 0.5
            # read the threshold csv
            with open(f'./thresholds_male/{self.attribute}_{self.model_name}.pkl', 'rb') as handle:
                self.thr = pickle.load(handle)
            with open(f'./thresholds_female/{self.attribute}_{self.model_name}.pkl', 'rb') as handle:
                self.thr_low = pickle.load(handle)
        elif self.attribute == 'race':
            self.coin = torch.randint(0, 5, (1,)).item()
            self.debias_race_counter = -1

            with open(f'./thresholds_white/{self.attribute}_{self.model_name}.pkl', 'rb') as handle:
                self.thr_white = pickle.load(handle)
            
            with open(f'./thresholds_black/{self.attribute}_{self.model_name}.pkl', 'rb') as handle:
                self.thr_black = pickle.load(handle)
            
            with open(f'./thresholds_latino/{self.attribute}_{self.model_name}.pkl', 'rb') as handle:
                self.thr_latino = pickle.load(handle)
            
            with open(f'./thresholds_asian/{self.attribute}_{self.model_name}.pkl', 'rb') as handle:
                self.thr_asian = pickle.load(handle)
            
            with open(f'./thresholds_indian/{self.attribute}_{self.model_name}.pkl', 'rb') as handle:
                self.thr_indian = pickle.load(handle)
        
        if self.strength < 0:
            raise ValueError('Negative values of strength are not supported')

        if steer_type in ('casteer', 'interpret'):
            if self.attribute == 'race':
                # we have multiple steering vectors here
                self.casteer_vectors = defaultdict(list)
                self.races = ["white", "black", "latino", "asian", "indian"]
                for source_concept, target_concept, race in zip(source_concepts, target_concepts, self.races):
                    casteer_concept_transforms = defaultdict(lambda: defaultdict(list))
                    for num_steer in source_concept:
                        for place_in_unet in source_concept[num_steer]:
                            for block_idx in range(len(source_concept[num_steer][place_in_unet])):
                                source_vector = source_concept[num_steer][place_in_unet][block_idx]
                                if target_concept is not None:
                                    target_vector = target_concept[num_steer][place_in_unet][block_idx]
                                else:
                                    target_vector = torch.zeros_like(source_vector)
                                steering_vector = source_vector - target_vector

                                if len(steering_vector.shape) == 1:
                                    steering_vector = steering_vector.unsqueeze(0)
                                steering_vector = convert_to_widest_dtype(steering_vector, device=self.device).unsqueeze(-1)
                                
                                res = self.strength * (steering_vector @ torch.linalg.pinv(steering_vector))
                                P = torch.eye(res.shape[1], dtype=res.dtype, device=self.device).unsqueeze(0) - res
                                
                                casteer_concept_transforms[num_steer][place_in_unet].append((steering_vector.squeeze(-1), P))
                    self.casteer_vectors[race].append(casteer_concept_transforms)
            else:
                # we only have one steering vector here
                self.casteer_vectors = []
                for source_concept, target_concept in zip(source_concepts, target_concepts):
                    casteer_concept_transforms = defaultdict(lambda: defaultdict(list))
                    for num_steer in source_concept:
                        for place_in_unet in source_concept[num_steer]:
                            for block_idx in range(len(source_concept[num_steer][place_in_unet])):
                                source_vector = source_concept[num_steer][place_in_unet][block_idx]
                                if target_concept is not None:
                                    target_vector = target_concept[num_steer][place_in_unet][block_idx]
                                else:
                                    target_vector = torch.zeros_like(source_vector)
                                steering_vector = source_vector - target_vector

                                if len(steering_vector.shape) == 1:
                                    steering_vector = steering_vector.unsqueeze(0)
                                steering_vector = convert_to_widest_dtype(steering_vector, device=self.device).unsqueeze(-1)
                                
                                res = self.strength * (steering_vector @ torch.linalg.pinv(steering_vector))
                                P = torch.eye(res.shape[1], dtype=res.dtype, device=self.device).unsqueeze(0) - res
                                
                                casteer_concept_transforms[num_steer][place_in_unet].append((steering_vector.squeeze(-1), P))
                    self.casteer_vectors.append(casteer_concept_transforms)
        else:
            raise ValueError(f'Unknown steer_type = {steer_type}')

        self.steering_cache = {}
        self.model_to_steer = model_to_steer

    def _convert_type(self, vector: torch.Tensor):
        return convert_to_widest_dtype(vector, device=self.device, force_double=False)

    def do_debias(self, vector: torch.Tensor, 
                *steering_tensors: torch.Tensor, 
                coin=True, 
                save_vectors=False, 
                save_vectors_path=None,
                diffusion_step=None,
                place_in_unet=None,
                block_index=None) -> torch.Tensor:
        assert len(vector.shape) == 4

        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        if self.attribute == 'race':
            b_list = list(steering_tensors)
            projection_scores_races = []
            b_norms_races = []
            for b in b_list:
                b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)
                b_norms_races += [b_norm]
                projection_scores = (
                    (
                        vector_reshaped
                    ) @ b_norm.unsqueeze(-1)
                ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)
                projection_scores_races += [projection_scores]
        else:
            (b,_) = steering_tensors
            b_norm_reshaped = b_norm.unsqueeze(-1)
            # computing dot products between vector components and steering vector x
            projection_scores = (
                (
                    vector_reshaped
                ) @ b_norm_reshaped
            ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)
        
        

        if save_vectors:
            assert self.attribute is None, "this is meant only for computing thresholds!"
            payload = {
                "scores": projection_scores.detach().cpu(),
                "diffusion_step": diffusion_step,
                "place_in_unet": place_in_unet,
                "block_index": block_index,
                "counter": self.counter,
            }
            os.makedirs(save_vectors_path, exist_ok=True)

            with open(os.path.join(save_vectors_path, f"projection_scores_{self.counter}.pkl"), "wb") as h:
                pickle.dump(payload, h)
            self.counter += 1
        
        # now, let's use the threshold for this particular block
        if self.llm:
            # perform debiasing through the llm -- to be implemented
            pass
        elif self.attribute == 'gender':
            # perform debiasing here
            max_thr = self.thr[(diffusion_step, place_in_unet, block_index)]['stats'] 
            min_thr = self.thr_low[(diffusion_step, place_in_unet, block_index)]['stats']
        elif self.attribute == 'race':
            max_thrs = [self.thr_white[(diffusion_step, place_in_unet, block_index)]['stats'] ,
                        self.thr_black[(diffusion_step, place_in_unet, block_index)]['stats'] ,
                        self.thr_latino[(diffusion_step, place_in_unet, block_index)]['stats'] ,
                        self.thr_asian[(diffusion_step, place_in_unet, block_index)]['stats'] ,
                        self.thr_indian[(diffusion_step, place_in_unet, block_index)]['stats'] 
                    ]
        else:
            # so everything falls into [min_thr, max_thr], and debiasing is always done
            min_thr = 0
            max_thr = 0


        if self.debias_type == 'discrete':
            if diffusion_step == 0 and place_in_unet == 'down' and block_index == 15:
                if self.attribute == 'gender':
                    projection_scores_max = projection_scores.max()
                    projection_scores_min = projection_scores.min()
                    print(projection_scores_min.item(), projection_scores_max.item(), max_thr, min_thr)
                    if (projection_scores_max < max_thr) and (projection_scores_min > min_thr):
                        self.debias = True
                elif self.attribute == 'race':
                    for race_counter, projection_scores_race  in enumerate(projection_scores_races):
                        projection_scores_race_max = projection_scores_race.max()
                        print(projection_scores_race_max.item(), max_thrs[race_counter])
                        if (projection_scores_race_max < max_thrs[race_counter]):
                            self.debias = True
                            # store counter only for debiasing
                            self.debias_race_counter = race_counter

                            # break

        else:
            self.debias = True
            
        if self.debias:
            if self.attribute == 'gender':
                steering_delta = - 1* projection_scores.to(vector.device) * b_norm.to(vector.device)
                vector_new = vector + steering_delta
                if coin:
                    vector_new = vector_new + torch.abs(projection_scores.to(vector.device)) * b_norm.to(vector.device)
                else:
                    vector_new = vector_new - torch.abs(projection_scores.to(vector.device)) * b_norm.to(vector.device)
            elif self.attribute == 'race':
                if self.debias_type == 'discrete':
                    steering_delta = -1* projection_scores_races[self.debias_race_counter].to(vector.device) * b_norms_races[self.debias_race_counter].to(vector.device)
                    vector_new = vector + steering_delta
                    vector_new = vector_new + torch.abs(projection_scores_races[coin].to(vector.device)) * b_norms_races[coin].to(vector.device)
                else:
                    # here, let's do an iterative debiasing
                    vector_new = None
                    for race_idx, race in enumerate(self.races):
                        # no steering back if the race is not present -- skip steering for negative delta?
                        steering_delta = -1 * projection_scores_races[race_idx].to(vector.device) * b_norms_races[race_idx].to(vector.device)
                        if vector_new is None:
                            vector_new = vector + steering_delta
                        else:
                            vector_new = vector_new + steering_delta
                    
                    # now the uniform distribution
                    if vector_new is None:
                        vector_new = vector + torch.abs(projection_scores_races[coin].to(vector.device)) * b_norms_races[coin].to(vector.device)
                    else:
                        vector_new = vector_new + torch.abs(projection_scores_races[coin].to(vector.device)) * b_norms_races[coin].to(vector.device)
                    
        else:
            vector_new = vector

        if self.debias_type != 'discrete':
            if self.attribute == 'gender':
                vector_new = torch.where(projection_scores < max_thr, vector_new, vector)
                vector_new = torch.where(projection_scores > min_thr, vector_new, vector)
            elif self.attribute == 'race':
                vector_new = torch.where(projection_scores_races[coin] < max_thrs[coin], vector_new, vector)
            
        return vector_new
    
    
    def renormalize(self, vector: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        if self.renormalize_after_steering:
            return vector / (torch.norm(vector, dim=-1, keepdim=True) + EPS) * norm
        else:
            return vector

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step: int, place_in_unet: str, block_index: int, min_token_index: int = None):
        batch_size = vector.shape[0]
        if batch_size > 1 and self.model_to_steer == ModelToSteer.UNET:
            # TODO: fix it properly sometime later
            # Steer only the prompt part of SDXL classifier-free guidance method
            batch_slice = slice(batch_size // 2, None)
            warnings.warn('Steering only the prompt part of SDXL classifier-free guidance (assumed the batch_idx=0 is not conditioned on the prompt)')
        else:
            batch_slice = slice(None, None)

        vector = vector.detach().clone()

        num_steer = 0 if self.use_first_diffusion_step else diffusion_step
        
        norm = torch.norm(vector, dim=-1, keepdim=True)
        for casteer_vectors in self.casteer_vectors:

            if self.attribute == 'race':
                # improve it later
                vecs_for_location = []
                for cv in self.casteer_vectors.values():
                    cv_dict = cv[0]
                    entry = cv_dict[num_steer][place_in_unet][block_index]
                    d = entry if torch.is_tensor(entry) else entry[0]
                    vecs_for_location.append(d)

                vector[batch_slice, ...] = self.do_debias(vector[batch_slice, ...], 
                                                    *vecs_for_location, 
                                                    coin=self.coin, 
                                                    save_vectors=self.save_vectors,
                                                    save_vectors_path = self.save_vectors_path,
                                                    diffusion_step=diffusion_step,
                                                    place_in_unet=place_in_unet,
                                                    block_index=block_index
                                                )
            else:
                vector[batch_slice, ...] = self.do_debias(vector[batch_slice, ...], 
                                                    *casteer_vectors[num_steer][place_in_unet][block_index], 
                                                    coin=self.coin, 
                                                    save_vectors=self.save_vectors,
                                                    save_vectors_path = self.save_vectors_path,
                                                    diffusion_step=diffusion_step,
                                                    place_in_unet=place_in_unet,
                                                    block_index=block_index
                                                )
            vector = self.renormalize(vector, norm)

        return vector.half()


