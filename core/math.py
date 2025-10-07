import typing as tp
import torch

def convert_to_widest_dtype(vector: torch.Tensor, device: tp.Any, force_double: bool = False):
    # float64 is needed for numerical stability
    if device.type == 'mps':
        if force_double:
            return vector.to('cpu').to(dtype=torch.float64)
        else:
            return vector.to(device, dtype=torch.float32)
    else:
        return vector.to(device, dtype=torch.float64)