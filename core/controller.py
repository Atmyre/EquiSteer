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


logger = logging.getLogger()

EPS = 1e-6

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
        mu_neutral: SteeringVectors | None,
        sigma_neutral: SteeringVectors | None,
        strength: float,

        mode: DiffusionVectorControlMode = None,
        mmsteer_vectors=None,
        steer_type: str = None,

        steer_only_up=False, 
        steer_back: bool = False,
        device: Any,
        num_layers: int = None,
        renormalize_after_steering: bool = False,
        intermediate_clipping: bool = True,
        use_first_diffusion_step: bool = False
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
        
        if self.strength < 0:
            raise ValueError('Negative values of strength are not supported')

        if steer_type in ('casteer', 'interpret'):
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

    def steer_transform(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        assert len(vector.shape) == 4
        (proj_left, proj_right, m_neutral) = steering_tensors

        num_heads = proj_left.shape[0]
        hidden_dim = proj_left.shape[2]
        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        assert vector.shape[2] == num_heads
        assert vector.shape[3] == hidden_dim

        # Assert proj_right dimensions for matrix multiplications
        assert proj_right.shape[0] == num_heads
        assert proj_right.shape[2] == proj_left.shape[1]
        assert proj_right.shape[1] == hidden_dim

        # Center the vector around m_neutral
        vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        m_neutral_expanded = m_neutral.to(vector.device).unsqueeze(1)
        vector_centered = vector_reshaped - m_neutral_expanded
        
        # Compute projection and apply steering: vector @ (I - strength * proj_right.mT @ proj_left.mT)
        projection_scores = vector_centered @ proj_right.to(vector.device)  # output shape = [num_heads, *, k]
        
        if self.intermediate_clipping:
            projection_scores = torch.where(projection_scores > 0, projection_scores, 0.0)
        
        steering_delta = -self.strength * (projection_scores @ proj_left.to(vector.device))  # output shape = [num_heads, *, hidden_dim]
        
        vector_steered = (vector_reshaped + steering_delta).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim)
        return vector_steered
    
    def steer_backward_CASteer_matrix_form(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        (_,P) = steering_tensors

        vector_steered = ((
            convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1) @ P.to(vector.device).mT)).transpose(0, 1).reshape(batch_size, sequence_length, num_heads, hidden_dim) 
        return vector_steered

    # steering backward, i.e. removing notion from vector
    def steer_backward_CASteer(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        assert len(vector.shape) == 4

        batch_size = vector.shape[0]
        sequence_length = vector.shape[1]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]
        (b,_) = steering_tensors

        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)

        vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(-1, num_heads, hidden_dim).transpose(0, 1)
        b_norm_reshaped = b_norm.unsqueeze(-1)
        
        # computing dot products between vector components and steering vector x
        projection_scores = (
            (
                vector_reshaped
            ) @ b_norm_reshaped
        ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)
        

        # we will steer back only if dot product is positive, i.e.
        # if there's positive amount of information from steering vector in the vector
        if self.intermediate_clipping:
            projection_scores = torch.where(projection_scores>0, projection_scores, 0)

        steering_delta = - self.strength * projection_scores.to(vector.device) * b_norm.to(vector.device)

        # steer backward for beta*sim
        return vector + steering_delta
    
    def interpret(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        (b,_) = steering_tensors
        b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)
        return b_norm.to(vector.device)

    def steer_forward_CASteer(self, vector: torch.Tensor, *steering_tensors: torch.Tensor) -> torch.Tensor:
        (b,_) = steering_tensors

        assert len(b.shape) in (1, 2)
        if len(b.shape) == 1:
            b = b.reshape(1, -1)

        return vector + self.strength * b.to(vector.device) * torch.norm(vector, dim=-1, keepdim=True).to(vector.device)
    
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

        if self.model_to_steer == ModelToSteer.LLAMA or (place_in_unet in ['up', 'mid', 'joint', 'single', 'sana'] or (place_in_unet == 'down' and not self.steer_only_up)):
            # if steering vectors are from turbo version, then there's only one key in self.steering_vectors, 
            # and we'll use it for all the steps of generation
            # if steering vectors are from full version, then there's a key in self.steering_vectors
            # for each of the generation steps 
            # TODO: general way to handle this
            num_steer = 0 if self.use_first_diffusion_step else diffusion_step

            norm = torch.norm(vector, dim=-1, keepdim=True)
            if self.steer_type == 'casteer':
                if self.steer_back:
                    for casteer_vectors in self.casteer_vectors:
                        vector[batch_slice, ...] = self.steer_backward_CASteer(vector[batch_slice, ...], *casteer_vectors[num_steer][place_in_unet][block_index])
                        vector = self.renormalize(vector, norm)
                else:
                    for casteer_vectors in self.casteer_vectors:
                        vector[batch_slice, ...] = self.steer_forward_CASteer(vector[batch_slice, ...], *casteer_vectors[num_steer][place_in_unet][block_index])
                        vector = self.renormalize(vector, norm)
            else:
                raise ValueError(f'Unknown steer type {self.steer_type}')
        return vector.half()




