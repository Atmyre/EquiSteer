import warnings
from core.pickle import pickle_stats
from core.controller import EPS, ModelToSteer, VectorControl, DiffusionVectorControlMode
from collections import defaultdict
import torch
import enum
from core.math import convert_to_widest_dtype


class TokenAggregationMode(enum.StrEnum):
    ALL = 'all'
    LAST = 'last'
    AVERAGE = 'average'

i = 0

class CrossAttentionOutputStatsCollector(VectorControl):
    def __init__(self,
                 mode: DiffusionVectorControlMode=None,
                 *,
                 token_aggregation_mode: TokenAggregationMode,
                 normalize: bool = False,
                 last_token_offset: int = -1
                ):
        super().__init__(mode=mode)

        self._cnt = defaultdict(lambda: defaultdict(list))
        self._m = defaultdict(lambda: defaultdict(list))  # running sum
        self._mm = defaultdict(lambda: defaultdict(list))  # running sum of squares
        self._s = defaultdict(lambda: defaultdict(list))  # running sum of squared differences (for Welford's algorithm)

        self._token_aggregation_mode = token_aggregation_mode
        self._last_token_offset = last_token_offset
        self._normalize = normalize
    
    def _update_statistics(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        stat_count = vector.shape[1]
        stat_m = torch.sum(vector, dim=1)

        if len(self._cnt[diffusion_step][place_in_unet]) <= block_index:
            self._cnt[diffusion_step][place_in_unet].append(stat_count)
            self._m[diffusion_step][place_in_unet].append(stat_m)
        else:
            self._cnt[diffusion_step][place_in_unet][block_index] += stat_count
            self._m[diffusion_step][place_in_unet][block_index] += stat_m
    
    def _update_statistics_welford(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index):
        # vector shape: [num_heads, num_samples, hidden_size]
        stat_count = vector.shape[1]
        stat_m = torch.sum(vector, dim=1)

        if len(self._cnt[diffusion_step][place_in_unet]) <= block_index:
            # Initialize new block
            self._cnt[diffusion_step][place_in_unet].append(stat_count)
            self._m[diffusion_step][place_in_unet].append(stat_m)
        else:
            # Update existing block using Welford's algorithm
            old_count = self._cnt[diffusion_step][place_in_unet][block_index]
            
            # Update count
            new_count = old_count + stat_count
            self._cnt[diffusion_step][place_in_unet][block_index] = new_count
            
            # Update mean using Welford's formula
            new_sum = self._m[diffusion_step][place_in_unet][block_index] + stat_m
            self._m[diffusion_step][place_in_unet][block_index] = new_sum

    # [batch_size, sequence_length, num_heads, head_dim]
    def forward(self, vector: torch.Tensor, diffusion_step, place_in_unet, block_index, min_token_index: int = None):
        batch_size = vector.shape[0]
        if batch_size > 1 and hasattr(self, "model_to_steer"):
            if self.model_to_steer == ModelToSteer.UNET:
                # TODO: fix it properly sometime later
                # Steer only the prompt part of SDXL classifier-free guidance method
                batch_slice = slice(batch_size // 2, None)
                warnings.warn('Collecting stats only for the prompt part of SDXL classifier-free guidance (assumed the batch_idx=0 is not conditioned on the prompt)')
        else:
            batch_slice = slice(None, None)
        
        
        num_heads = vector.shape[-2]
        hidden_size = vector.shape[-1]

        if min_token_index is not None and vector.shape[1] > 1:
            vector_slices = vector[batch_slice, min_token_index:, :, :]
        else:
            vector_slices = vector[batch_slice, ...]


        vector_permuted = vector_slices.permute(2, 0, 1, 3)  # [num_heads, batch_size, sequence_length, head_dim]
        vec = convert_to_widest_dtype(vector_permuted.view(num_heads, -1, hidden_size), device=vector_permuted.device)
        if self._token_aggregation_mode == TokenAggregationMode.AVERAGE:
            vec = torch.mean(vec, dim=1, keepdim=True)
        elif self._token_aggregation_mode == TokenAggregationMode.LAST:
            if batch_size > 1:
                raise ValueError("TokenAggregationMode.LAST and batch_size > 1 is not supported currently")
            start = self._last_token_offset
            end = self._last_token_offset + 1
            if end == 0:
                end = vec.shape[1]
            vec = vec[:, start:end, :]
            assert vec.shape[1] == 1

        if self._normalize:
            vec /= torch.linalg.norm(vec, dim=2, keepdim=True) + EPS

        self._update_statistics_welford(vec, diffusion_step, place_in_unet, block_index)
        
        return vector
    
    @property
    def means(self):
        result = {}
        for diffusion_step in self._m:
            result[diffusion_step] = {}
            for place_in_unet in self._m[diffusion_step]:
                result[diffusion_step][place_in_unet] = []
                for block_idx in range(len(self._m[diffusion_step][place_in_unet])):
                    count = self._cnt[diffusion_step][place_in_unet][block_idx]
                    m = self._m[diffusion_step][place_in_unet][block_idx] / count
                    result[diffusion_step][place_in_unet].append(m)
        return result

    def save_stats(self, *, means_path: str = None, use_torch_save: bool = False):
        if means_path is not None:
            if use_torch_save:
                torch.save(self.means, means_path)
            else:
                pickle_stats(self.means, means_path)
