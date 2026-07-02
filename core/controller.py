import logging
import warnings
import numpy as np
import torch
import abc
import typing as tp
from collections import defaultdict
from typing import Optional, Any
from core.pickle import unpickle
import torch.nn.functional as F

import enum
from core.math import convert_to_widest_dtype
import pickle
import os

logger = logging.getLogger()

EPS = 1e-6


def _resolve_path(rel_path: str) -> str:
    """Resolve a steering-vector / threshold path.

    If env var FAIRSTEER_ROOT is set, look only under that root (no fallback —
    explicit user choice). Otherwise try the default repo bases in order.
    """
    override = os.environ.get("FAIRSTEER_ROOT")
    if override:
        p = f"{override.rstrip('/')}/{rel_path}"
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{p} (FAIRSTEER_ROOT set; no fallback)")
        return p
    for base in ("/data/home/acw685/CA_diffusion_debiasing-main",
                 "/gpfs/scratch/acw685/CA_diffusion_debiasing-main"):
        p = f"{base}/{rel_path}"
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(rel_path)


class DiffusionVectorControlMode(enum.StrEnum):
    ATTN_OUTPUT = 'attn_output'
    ATTN_HEADS = 'attn_head'


class ModelToSteer(enum.StrEnum):
    UNET = 'unet'
    LLAMA = 'llama'


class VectorControl(abc.ABC):
    def __init__(self, mode: DiffusionVectorControlMode = None, num_layers: int = None):
        self._mode = mode
        self._active = True
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        self.num_attn_layers = num_layers
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
        # source_concepts, target_concepts =self._flip_coin()
        # self._generate_casteer_vectors(source_concepts, target_concepts)
        # self.debias = False
    
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
        intermediate_clipping: bool = True,
        use_first_diffusion_step: bool = False,
        save_vectors: bool = False,
        save_vectors_path: str = None,
        attribute: str = None,
        llm: bool = False, # for llm based decision for debiasing -- independent of threshold
        do_debias=True,
        do_erase=True,
        do_threshold=True,
        gate_threshold_multiplier: float = 1.0,
        # debias_type: str = 'discrete',
        model_name: str = 'sd15',
    ):
        super().__init__(mode=mode, num_layers=num_layers)
        self.device = device
        
        self.steer_only_up = steer_only_up
        self.steer_back = steer_back
        self.steer_type = steer_type
        self.renormalize_after_steering = renormalize_after_steering
        self.intermediate_clipping = intermediate_clipping
        self.strength = strength
        self.use_first_diffusion_step = use_first_diffusion_step
        self.save_vectors = save_vectors
        self.save_vectors_path = save_vectors_path
        self.attribute = attribute
        self.llm = llm
        self.counter = 0 # for saving in forward call
        self.model_name = model_name
        # Multiplier applied to the gating threshold (Eq. 5 in the paper) only.
        # Injection magnitude (Eq. 8) is left unchanged so that this knob isolates
        # the sensitivity of the gate, not the steering strength.
        self.gate_threshold_multiplier = float(gate_threshold_multiplier)

        print("ATTRIBUTE", self.attribute)
        if self.attribute in ['race', 'gender', 'glasses', 'age', 'body']:
            print("Doing debiasing")
            self.do_debias = True
            self.do_erase = True
            self.do_threshold = True
        else:
            self.do_debias = False
            self.do_erase = False
            self.do_threshold = False

        print(self.do_debias)
        if self.do_debias:
            print('loading steering vectors for', self.attribute)
            source_concepts, target_concepts = self._flip_coin()
        self.casteer_vectors = self._generate_casteer_vectors(source_concepts, target_concepts)
        
        # self.do_debias = do_debias
        # self.do_erase = do_erase
        # self.do_threshold = do_threshold
        
        # self.debias = False
        # self.debias_type = debias_type
        
        if self.strength < 0:
            raise ValueError('Negative values of strength are not supported')

        self.steering_cache = {}
        self.model_to_steer = model_to_steer

    def _generate_casteer_vectors(self, source_concepts, target_concepts):
        casteer_vectors = []
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

                        P = None
                        casteer_concept_transforms[num_steer][place_in_unet].append((steering_vector.squeeze(-1), P))
            casteer_vectors.append(casteer_concept_transforms)
        return casteer_vectors

    def _flip_coin(self):
        self.coin = torch.rand(1) 
        self._load_thresholds(self.attribute)
        return self._load_steering_vectors(self.attribute)

    def _load_steering_vectors_races(self):
        self.steering_vectors = []
        source_concepts = []
        target_concepts = []
        for race_to in ['White', 'Black', 'Asian', 'Indian', 'Latino']:
            source_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/race/{race_to}/pos_means_210.pickle')))
            target_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/race/{race_to}/neg_means_210.pickle')))
        return source_concepts, target_concepts

    def _load_steering_vectors_gender(self):
        self.steering_vectors = []
        source_concepts = []
        target_concepts = []
        for gender in ['male_female', 'female_male']:
            source_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/gender/{gender}/pos_means_126.pickle')))
            target_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/gender/{gender}/neg_means_126.pickle')))
        return source_concepts, target_concepts

    def _load_steering_vectors_age(self):
        self.steering_vectors = []
        source_concepts = []
        target_concepts = []
        for age_to in ['young', 'middle_aged', 'elderly']:
            source_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/age/{age_to}/pos_means_210.pickle')))
            target_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/age/{age_to}/neg_means_210.pickle')))
        return source_concepts, target_concepts

    def _load_steering_vectors_body(self):
        self.steering_vectors = []
        source_concepts = []
        target_concepts = []
        for body_to in ['slim', 'average', 'heavy']:
            source_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/body/{body_to}/pos_means_210.pickle')))
            target_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/body/{body_to}/neg_means_210.pickle')))
        return source_concepts, target_concepts

    def _load_steering_vectors_glasses(self):
        self.steering_vectors = []
        source_concepts = []
        target_concepts = []
        for gender in ['eyeglasses']:
            source_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/{gender}/pos_means_210.pickle')))
            target_concepts.append(unpickle(_resolve_path(f'steering_vectors/{self.model_name}/{gender}/neg_means_210.pickle')))
        return source_concepts, target_concepts

    def _load_steering_vectors(self, attribute: str):
        if attribute == 'race':
            source_concepts, target_concepts = self._load_steering_vectors_races()
        elif attribute == 'gender':
            source_concepts, target_concepts = self._load_steering_vectors_gender()
        elif attribute == 'glasses':
            source_concepts, target_concepts = self._load_steering_vectors_glasses()
        elif attribute == 'age':
            source_concepts, target_concepts = self._load_steering_vectors_age()
        elif attribute == 'body':
            source_concepts, target_concepts = self._load_steering_vectors_body()
        else:
            raise ValueError(f'Invalid attribute: {attribute}')

        return source_concepts, target_concepts

    def _load_thresholds_races(self):
        self.thr = []
        for race_to in ['White', 'Black', 'Asian', 'Indian', 'Latino']:
            self.thr.append(unpickle(_resolve_path(f'thresholds_race/{self.model_name}/{race_to}/race_{race_to}_{self.model_name}.pkl')))

    def _load_thresholds_gender(self):
        self.thr = []
        for gender in ['male_female', 'female_male']:
            self.thr.append(unpickle(_resolve_path(f'thresholds/wtf_sdxl/{self.model_name}/{gender}/gender_{gender}_{self.model_name}.pkl')))

    def _load_thresholds_age(self):
        self.thr = []
        for age_to in ['young', 'middle_aged', 'elderly']:
            self.thr.append(unpickle(_resolve_path(f'thresholds_age/{self.model_name}/{age_to}/age_{age_to}_{self.model_name}.pkl')))

    def _load_thresholds_body(self):
        self.thr = []
        for body_to in ['slim', 'average', 'heavy']:
            self.thr.append(unpickle(_resolve_path(f'thresholds_body/{self.model_name}/{body_to}/body_{body_to}_{self.model_name}.pkl')))

    def _load_thresholds_glasses(self):
        self.thr = []
        for gender in ['eyeglasses']:
            self.thr.append(unpickle(_resolve_path(f'thresholds/eyeglasses/{self.model_name}/{gender}/concepts_{gender}_{self.model_name}.pkl')))

    def _load_thresholds(self, attribute: str):
        if attribute == 'race':
            self._load_thresholds_races()
        elif attribute == 'gender':
            self._load_thresholds_gender()
        elif attribute == 'glasses':
            self._load_thresholds_glasses()
        elif attribute == 'age':
            self._load_thresholds_age()
        elif attribute == 'body':
            self._load_thresholds_body()
        else:
            raise ValueError(f'Invalid attribute: {attribute}')

    def _convert_type(self, vector: torch.Tensor):
        return convert_to_widest_dtype(vector, device=self.device, force_double=False)

    def debiasing(self, vector: torch.Tensor, 
                steering_tensors: Any, 
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
        projection_scores = []
        b_norms = []

        for steering_tensor in steering_tensors:
            (b,_) = steering_tensor
            b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)
            b_norms.append(b_norm)
            vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
            b_norm_reshaped = b_norm.unsqueeze(-1)
            projection_scores.append((
                (
                    vector_reshaped
                ) @ b_norm_reshaped
            ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)
            )


        if save_vectors:
            payload = {
                "scores": projection_scores[0].detach().cpu(),
                "diffusion_step": diffusion_step,
                "place_in_unet": place_in_unet,
                "block_index": block_index,
                "counter": self.counter,
            }
            os.makedirs(save_vectors_path, exist_ok=True)

            with open(os.path.join(save_vectors_path, f"projection_scores_{self.counter}.pkl"), "wb") as h:
                pickle.dump(payload, h)
            self.counter += 1
            return vector

        
        # now, let's use the threshold for this particular block
        if self.llm:
            # perform debiasing through the llm -- to be implemented
            pass
        elif self.do_debias:
            # perform debiasing here
            thrs = [x[(diffusion_step, place_in_unet, block_index)]['stats'] for x in self.thr]
            # FAIRSTEER_ADDBACK_FIELD env var controls which field of the threshold
            # pickle is used for the add-back magnitude (paper Eq. 8 alpha).
            # Default 'stats' = the gate threshold (legacy behaviour).
            # Set to 'mean_male_max' for the proper Eq. 8 (mean of maximal dot
            # products on attribute-specific prompts).
            addback_field = os.environ.get('FAIRSTEER_ADDBACK_FIELD', 'stats')
            if addback_field == 'stats':
                thrs_to_add = thrs
            else:
                thrs_to_add = [x[(diffusion_step, place_in_unet, block_index)][addback_field] for x in self.thr]
        else:
            # so everything falls into [min_thr, max_thr], and debiasing is never done
            thrs = [0]*len(projection_scores)


        print(self.do_debias, self.do_threshold, self.model_name, diffusion_step, block_index)
        if self.do_debias and self.do_threshold:
            if self.model_name in ['sd15', 'sd21']:
                needed_block_idx = 4
            elif self.model_name in ['sdxl']:
                needed_block_idx = 17
            elif self.model_name in ['sana15']:
                needed_block_idx = 5

            condition_sd = (self.model_name in ['sd15', 'sd21', 'sdxl'] and diffusion_step == 0 and place_in_unet == 'down' and block_index == needed_block_idx)
            condition_sana = (self.model_name in ['sana15'] and diffusion_step == 0 and block_index == needed_block_idx)
            
            if condition_sd or condition_sana:
                self.debias = True
                for thr, projection_score in zip(thrs, projection_scores):
                    projection_scores_max = projection_score.max()
                    gate_thr = thr * self.gate_threshold_multiplier
                    print(self.coin, projection_scores_max.item(), gate_thr, '(thr*mult)')
                    if (projection_scores_max > gate_thr):
                        self.debias = False
                print('------------------', self.debias)
        elif self.do_debias:
            self.debias = True
        else:
            self.debias = False

        diffusion_step_max = {
            'sd15': 20,
            'sd21': 20,
            'sdxl': 51,
            'sana15': 51,
        }
            
        if self.debias and diffusion_step < diffusion_step_max[self.model_name]:
            if self.do_erase:
                b_norm_prevs = []
                for b_norm in b_norms:

                    vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
                    b_norm_to_subtract = b_norm.clone()
                    
                    for b_norm_prev in b_norm_prevs:
                        b_norm_to_subtract = b_norm_to_subtract - (b_norm_to_subtract @ b_norm_prev.T) @ b_norm_prev
                        b_norm_to_subtract = b_norm_to_subtract / (torch.linalg.norm(b_norm_to_subtract, dim=-1, keepdim=True) + EPS)
                    
                    b_norm_reshaped = b_norm_to_subtract.unsqueeze(-1)
                    projection_score = (
                        (
                            vector_reshaped
                        ) @ b_norm_reshaped
                    ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)

                    steering_delta = - 1.0 * projection_score.to(vector.device) * b_norm_to_subtract.to(vector.device)
                    vector = vector+steering_delta

                    b_norm_prevs.append(b_norm_to_subtract)

            multiplier = 1.0
            # if self.do_erase: 
            #     multiplier = 1.0
            # else:
            #     multiplier = 0.5
            # Use thrs_to_add (set by FAIRSTEER_ADDBACK_FIELD) consistently for all
            # attributes' add-back magnitude. Falls back to thrs (gate threshold)
            # when the env var is unset.
            if self.attribute == 'race':
                if self.coin < 0.2:
                    vector = vector + multiplier*thrs_to_add[0]*b_norms[0]
                elif self.coin < 0.4:
                    vector = vector + multiplier*thrs_to_add[1]*b_norms[1]
                elif self.coin < 0.6:
                    vector = vector + multiplier*thrs_to_add[2]*b_norms[2]
                elif self.coin < 0.8:
                    vector = vector + multiplier*thrs_to_add[3]*b_norms[3]
                else:
                    vector = vector + multiplier*thrs_to_add[4]*b_norms[4]
            elif self.attribute == 'gender':
                if self.coin < 0.5:
                    vector = vector + multiplier*thrs_to_add[0]*b_norms[0]
                else:
                    vector = vector + multiplier*thrs_to_add[1]*b_norms[1]
            elif self.attribute == 'age':
                if self.coin < 1.0/3.0:
                    vector = vector + multiplier*thrs_to_add[0]*b_norms[0]
                elif self.coin < 2.0/3.0:
                    vector = vector + multiplier*thrs_to_add[1]*b_norms[1]
                else:
                    vector = vector + multiplier*thrs_to_add[2]*b_norms[2]
            elif self.attribute == 'body':
                if self.coin < 1.0/3.0:
                    vector = vector + multiplier*thrs_to_add[0]*b_norms[0]
                elif self.coin < 2.0/3.0:
                    vector = vector + multiplier*thrs_to_add[1]*b_norms[1]
                else:
                    vector = vector + multiplier*thrs_to_add[2]*b_norms[2]
            elif self.attribute in ['glasses']:
                if self.coin < 0.5:
                    vector = vector + multiplier*thrs_to_add[0]*b_norms[0]
                else:
                    vector = vector+steering_delta
            
        return vector
    
    
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
        # for casteer_vectors in self.casteer_vectors:
        steering_vectors = [casteer_vectors[num_steer][place_in_unet][block_index] for casteer_vectors in self.casteer_vectors]
        vector[batch_slice, ...] = self.debiasing(vector[batch_slice, ...], 
                                            steering_vectors,
                                            save_vectors=self.save_vectors,
                                            save_vectors_path = self.save_vectors_path,
                                            diffusion_step=diffusion_step,
                                            place_in_unet=place_in_unet,
                                            block_index=block_index
                                        )
        vector = self.renormalize(vector, norm)

        return vector.half()

    def reset(self):
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        self.coin = torch.rand(1) 
        # if self.attribute:
        #     source_concepts, target_concepts = self._flip_coin()
        #     self._generate_casteer_vectors(source_concepts, target_concepts)
        self.debias = False


