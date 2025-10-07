import os
from typing import Any
import torch
import pickle
import io


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def unpickle(path: str | None):
    if path is None:
        return None
    try:
        return torch.load(path, weights_only=False)
    except:
        try:
            with open(path, 'rb') as fin:
                return pickle.load(fin)
        except:
            with open(path, 'rb') as fin:
                return CPU_Unpickler(fin).load()


def unpickle_pack(path: str | None) -> list[dict]:
    if path is None:
        return None
    result = []
    for subpath in path.split(','):
        result.append(unpickle(subpath))
    return result


def pickle_stats(obj: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as fout:
        pickle.dump(obj, fout)