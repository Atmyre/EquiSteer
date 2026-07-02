import typing as tp
from collections import defaultdict
from typing import Any

import torch

from core.math import convert_to_widest_dtype
from core.pickle import unpickle
from .controller import (
    EPS,
    DiffusionVectorControlMode,
    ModelToSteer,
    SteeringVectors,
    VectorControl,
    _resolve_path,
)


def _to_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


class CrossAttentionOutputSteeringMulti(VectorControl):
    """
    Vector controller that supports simultaneous debiasing for race and gender.

    The controller loads race and gender steering vectors/thresholds and applies both
    transformations sequentially for each attention block.
    """

    def __init__(
        self,
        model_to_steer: ModelToSteer,
        *,
        strength: float,
        device: Any,
        model_name: str = "sd15",
        attributes: tp.Sequence[str] = ("race", "gender"),
        mode: DiffusionVectorControlMode = None,
        num_layers: int = None,
        steer_only_up: bool = False,
        steer_back: bool = False,
        renormalize_after_steering: bool = False,
        intermediate_clipping: bool = True,
        use_first_diffusion_step: bool = False,
        do_debias: Any = True,
        do_erase: Any = True,
        do_threshold: Any = True,
    ):
        super().__init__(mode=mode, num_layers=num_layers)
        self.device = device
        self.model_to_steer = model_to_steer
        self.model_name = model_name

        self.steer_only_up = steer_only_up
        self.steer_back = steer_back
        self.renormalize_after_steering = renormalize_after_steering
        self.intermediate_clipping = intermediate_clipping
        self.use_first_diffusion_step = use_first_diffusion_step

        self.attributes = [a.lower() for a in attributes]
        for attribute in self.attributes:
            if attribute not in ("race", "gender", "age", "body"):
                raise ValueError(
                    f"Unsupported attribute: {attribute}. Supported: race, gender, age, body."
                )

        self.do_debias = _to_bool(do_debias, default=True)
        self.do_erase = _to_bool(do_erase, default=True)
        self.do_threshold = _to_bool(do_threshold, default=True)

        if strength < 0:
            raise ValueError("Negative values of strength are not supported")
        self.strength = strength

        # Per-attribute random coin for target balancing.
        self.coins: dict[str, torch.Tensor] = {}
        self.thr: dict[str, list] = {}
        self.casteer_vectors: dict[str, list] = {}

        self._initialize_attribute_state()

    def _initialize_attribute_state(self):
        for attribute in self.attributes:
            self.coins[attribute] = torch.rand(1)
            self.thr[attribute] = self._load_thresholds(attribute)
            source_concepts, target_concepts = self._load_steering_vectors(attribute)
            self.casteer_vectors[attribute] = self._generate_casteer_vectors(
                source_concepts, target_concepts
            )

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
                        casteer_concept_transforms[num_steer][place_in_unet].append(
                            (steering_vector.squeeze(-1), None)
                        )
            casteer_vectors.append(casteer_concept_transforms)
        return casteer_vectors

    def _load_steering_vectors_race(self):
        source_concepts = []
        target_concepts = []
        for race_to in ["White", "Black", "Asian", "Indian", "Latino"]:
            source_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/race/{race_to}/pos_means_210.pickle")))
            target_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/race/{race_to}/neg_means_210.pickle")))
        return source_concepts, target_concepts

    def _load_steering_vectors_gender(self):
        source_concepts = []
        target_concepts = []
        for gender in ["male_female", "female_male"]:
            source_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/gender/{gender}/pos_means_126.pickle")))
            target_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/gender/{gender}/neg_means_126.pickle")))
        return source_concepts, target_concepts

    def _load_steering_vectors_age(self):
        source_concepts = []
        target_concepts = []
        for age_to in ["young", "middle_aged", "elderly"]:
            source_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/age/{age_to}/pos_means_210.pickle")))
            target_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/age/{age_to}/neg_means_210.pickle")))
        return source_concepts, target_concepts

    def _load_steering_vectors_body(self):
        source_concepts = []
        target_concepts = []
        for body_to in ["slim", "average", "heavy"]:
            source_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/body/{body_to}/pos_means_210.pickle")))
            target_concepts.append(unpickle(_resolve_path(f"steering_vectors/{self.model_name}/body/{body_to}/neg_means_210.pickle")))
        return source_concepts, target_concepts

    def _load_steering_vectors(self, attribute: str):
        if attribute == "race":
            return self._load_steering_vectors_race()
        if attribute == "gender":
            return self._load_steering_vectors_gender()
        if attribute == "age":
            return self._load_steering_vectors_age()
        if attribute == "body":
            return self._load_steering_vectors_body()
        raise ValueError(f"Unsupported attribute: {attribute}")

    def _load_thresholds_race(self):
        thresholds = []
        for race_to in ["White", "Black", "Asian", "Indian", "Latino"]:
            thresholds.append(unpickle(_resolve_path(f"thresholds_race/{self.model_name}/{race_to}/race_{race_to}_{self.model_name}.pkl")))
        return thresholds

    def _load_thresholds_gender(self):
        thresholds = []
        for gender in ["male_female", "female_male"]:
            thresholds.append(unpickle(_resolve_path(f"thresholds/wtf_sdxl/{self.model_name}/{gender}/gender_{gender}_{self.model_name}.pkl")))
        return thresholds

    def _load_thresholds_age(self):
        thresholds = []
        for age_to in ["young", "middle_aged", "elderly"]:
            thresholds.append(unpickle(_resolve_path(f"thresholds_age/{self.model_name}/{age_to}/age_{age_to}_{self.model_name}.pkl")))
        return thresholds

    def _load_thresholds_body(self):
        thresholds = []
        for body_to in ["slim", "average", "heavy"]:
            thresholds.append(unpickle(_resolve_path(f"thresholds_body/{self.model_name}/{body_to}/body_{body_to}_{self.model_name}.pkl")))
        return thresholds

    def _load_thresholds(self, attribute: str):
        if attribute == "race":
            return self._load_thresholds_race()
        if attribute == "gender":
            return self._load_thresholds_gender()
        if attribute == "age":
            return self._load_thresholds_age()
        if attribute == "body":
            return self._load_thresholds_body()
        raise ValueError(f"Unsupported attribute: {attribute}")

    def _get_thrs_for_block(self, attribute: str, diffusion_step: int, place_in_unet: str, block_index: int, field: str = "stats"):
        if not self.do_debias:
            return []
        return [
            x[(diffusion_step, place_in_unet, block_index)][field] for x in self.thr[attribute]
        ]

    def _compute_projection_scores(self, vector: torch.Tensor, steering_tensors: list):
        batch_size = vector.shape[0]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]

        projection_scores = []
        b_norms = []
        for steering_tensor in steering_tensors:
            b, _ = steering_tensor
            b_norm = b / torch.linalg.norm(b, dim=-1, keepdim=True)
            b_norms.append(b_norm)

            vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(
                -1, num_heads, hidden_dim
            ).transpose(0, 1)
            b_norm_reshaped = b_norm.unsqueeze(-1)
            projection_score = (
                vector_reshaped @ b_norm_reshaped
            ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)
            projection_scores.append(projection_score)
        return projection_scores, b_norms

    def _threshold_gate(self, projection_scores: list, thrs: list, diffusion_step: int, place_in_unet: str, block_index: int):
        if not self.do_threshold:
            return True

        if self.model_name in ["sd15", "sd21"]:
            needed_block_idx = 4
        elif self.model_name == "sdxl":
            needed_block_idx = 17
        elif self.model_name == "sana15":
            needed_block_idx = 5
        else:
            needed_block_idx = -1

        condition_sd = (
            self.model_name in ["sd15", "sd21", "sdxl"]
            and diffusion_step == 0
            and place_in_unet == "down"
            and block_index == needed_block_idx
        )
        condition_sana = (
            self.model_name in ["sana15"]
            and diffusion_step == 0
            and block_index == needed_block_idx
        )

        if not (condition_sd or condition_sana):
            return True

        for thr, projection_score in zip(thrs, projection_scores):
            if projection_score.max() > thr:
                return False
        return True

    def _erase_component(self, vector: torch.Tensor, b_norms: list):
        batch_size = vector.shape[0]
        num_heads = vector.shape[2]
        hidden_dim = vector.shape[3]

        b_norm_prevs = []
        for b_norm in b_norms:
            vector_reshaped = convert_to_widest_dtype(vector, device=self.device).reshape(
                -1, num_heads, hidden_dim
            ).transpose(0, 1)
            b_norm_to_subtract = b_norm.clone()
            for b_norm_prev in b_norm_prevs:
                b_norm_to_subtract = b_norm_to_subtract - (
                    (b_norm_to_subtract @ b_norm_prev.T) @ b_norm_prev
                )
                b_norm_to_subtract = b_norm_to_subtract / (
                    torch.linalg.norm(b_norm_to_subtract, dim=-1, keepdim=True) + EPS
                )

            b_norm_reshaped = b_norm_to_subtract.unsqueeze(-1)
            projection_score = (
                vector_reshaped @ b_norm_reshaped
            ).transpose(0, 1).reshape(batch_size, -1, num_heads, 1)

            steering_delta = -1.0 * projection_score.to(vector.device) * b_norm_to_subtract.to(vector.device)
            vector = vector + steering_delta
            b_norm_prevs.append(b_norm_to_subtract)
        return vector

    def _add_back(self, attribute: str, vector: torch.Tensor, b_norms: list, thrs: list):
        multiplier = 1.0
        coin = self.coins[attribute]
        if attribute == "race":
            if coin < 0.2:
                return vector + multiplier * thrs[0] * b_norms[0]
            if coin < 0.4:
                return vector + multiplier * thrs[1] * b_norms[1]
            if coin < 0.6:
                return vector + multiplier * thrs[2] * b_norms[2]
            if coin < 0.8:
                return vector + multiplier * thrs[3] * b_norms[3]
            return vector + multiplier * thrs[4] * b_norms[4]
        if attribute == "gender":
            if coin < 0.5:
                return vector + multiplier * thrs[0] * b_norms[0]
            return vector + multiplier * thrs[1] * b_norms[1]
        if attribute == "age":
            if coin < 1.0 / 3.0:
                return vector + multiplier * thrs[0] * b_norms[0]
            if coin < 2.0 / 3.0:
                return vector + multiplier * thrs[1] * b_norms[1]
            return vector + multiplier * thrs[2] * b_norms[2]
        if attribute == "body":
            if coin < 1.0 / 3.0:
                return vector + multiplier * thrs[0] * b_norms[0]
            if coin < 2.0 / 3.0:
                return vector + multiplier * thrs[1] * b_norms[1]
            return vector + multiplier * thrs[2] * b_norms[2]
        raise ValueError(f"Unsupported attribute: {attribute}")

    def _debias_attribute(
        self,
        vector: torch.Tensor,
        attribute: str,
        steering_tensors: list,
        diffusion_step: int,
        place_in_unet: str,
        block_index: int,
    ):
        projection_scores, b_norms = self._compute_projection_scores(vector, steering_tensors)
        thrs = self._get_thrs_for_block(attribute, diffusion_step, place_in_unet, block_index, field="stats")
        if not self.do_debias:
            return vector
        if not self._threshold_gate(projection_scores, thrs, diffusion_step, place_in_unet, block_index):
            return vector

        if self.do_erase:
            vector = self._erase_component(vector, b_norms)

        # Use FAIRSTEER_ADDBACK_FIELD env var to select the add-back magnitude
        # ('stats' = legacy gate-threshold midpoint, 'mean_male_max' = paper Eq.8).
        import os as _os
        addback_field = _os.environ.get("FAIRSTEER_ADDBACK_FIELD", "stats")
        if addback_field != "stats":
            thrs = self._get_thrs_for_block(attribute, diffusion_step, place_in_unet, block_index, field=addback_field)

        diffusion_step_max = {
            "sd15": 20,
            "sd21": 20,
            "sdxl": 51,
            "sana15": 51,
        }
        max_step = diffusion_step_max.get(self.model_name, 51)
        if diffusion_step < max_step:
            vector = self._add_back(attribute, vector, b_norms, thrs)
        return vector

    def renormalize(self, vector: torch.Tensor, norm: torch.Tensor) -> torch.Tensor:
        if self.renormalize_after_steering:
            return vector / (torch.norm(vector, dim=-1, keepdim=True) + EPS) * norm
        return vector

    def forward(
        self,
        vector: torch.Tensor,
        diffusion_step: int,
        place_in_unet: str,
        block_index: int,
        min_token_index: int = None,
    ):
        batch_size = vector.shape[0]
        if batch_size > 1 and self.model_to_steer == ModelToSteer.UNET:
            batch_slice = slice(batch_size // 2, None)
        else:
            batch_slice = slice(None, None)

        vector = vector.detach().clone()
        norm = torch.norm(vector, dim=-1, keepdim=True)
        num_steer = 0 if self.use_first_diffusion_step else diffusion_step

        for attribute in self.attributes:
            steering_vectors = [
                casteer[num_steer][place_in_unet][block_index]
                for casteer in self.casteer_vectors[attribute]
            ]
            vector[batch_slice, ...] = self._debias_attribute(
                vector[batch_slice, ...],
                attribute=attribute,
                steering_tensors=steering_vectors,
                diffusion_step=diffusion_step,
                place_in_unet=place_in_unet,
                block_index=block_index,
            )

        vector = self.renormalize(vector, norm)
        return vector.half()

    def reset(self):
        self._diffusion_step = 0
        self._current_attn_layer = 0
        self._current_position = defaultdict(int)
        for attribute in self.attributes:
            self.coins[attribute] = torch.rand(1)
        self.debias = False
