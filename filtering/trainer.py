import os
import json
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import Sampler

from transformers import (
    Trainer,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
)
from transformers.trainer import (
    is_sagemaker_mp_enabled,
    get_parameter_names,
    has_length,
    ALL_LAYERNORM_LAYERS,
    logger,
)


class LengthGroupedSampler(Sampler):
    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")
        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


class Qwen2VLForConditionalGeneration_SelfFilter(Qwen2VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.score_net = nn.Linear(1024, 1, bias=False)

    def predict_weights(self, scores: torch.Tensor) -> torch.Tensor:
        
        return self.score_net(scores)

    def get_score_net_dtype(self):

        return next(self.score_net.parameters()).dtype


ALL_LAYERNORM_LAYERS = ["norm", "layernorm", "layer_norm"]

class Qwen2VLTrainer_SelfFilter(Trainer):

    def __init__(self, *args, **kwargs):
        self.feature_extractor_setting = kwargs.pop('feature_extractor_setting', None)
        super().__init__(*args, **kwargs)

        # clip feature or multi-scores(json) 로드
        if self.feature_extractor_setting == 'clip':
            self.clip_feat = torch.load(self.args.clip_feature_path)
        elif self.feature_extractor_setting == 'scores':
            score_names = [
                "/root/vol1/cocodataset/llava_clipscore.json",
                "/root/vol1/cocodataset/llava_imagereward.json",
            ]
            self.score_dicts = self._load_scores(score_names)
        else:
            print('Unknown feature_extractor_setting: ', self.feature_extractor_setting)
            raise NotImplementedError

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        """
        length-based sampler or default
        """
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if getattr(self.args, 'group_by_modality_length', False):
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                self.args.train_batch_size,
                world_size=self.args.world_size * self.args.gradient_accumulation_steps,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):

        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model
        if self.optimizer is None:

            decay_parameters = get_parameter_names(opt_model, (nn.LayerNorm, ))
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def _save_checkpoint(self, model, trial, metrics=None):
        pass
        


    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        super()._save(output_dir, state_dict)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):

        unique_indices = inputs.pop("unique_indices", None)
        loss, outputs = super().compute_loss(model, inputs, return_outputs=True)


        labels = inputs["labels"]
        logits = outputs.logits

        loss = self._get_weighted_loss(model, unique_indices, logits, labels)
        return (loss, outputs) if return_outputs else loss

    def _load_scores(self, score_names: List[str]):
        def norm_scores(score_dict: dict):
            min_score = min(score_dict.values())
            max_score = max(score_dict.values())
            normed_score_dict = {
                k: (v - min_score) / (max_score - min_score) * 2 - 1
                for k, v in score_dict.items()
            }
            return normed_score_dict

        score_dicts = []
        for score_name in score_names:
            with open(score_name, "r") as f:
                score_dict = json.load(f)
                score_dicts.append(norm_scores(score_dict))
        return score_dicts

    def _get_weighted_loss(self, model, unique_indices, logits, labels):
        if hasattr(model, "module"):  
            model = model.module 

        weights = self._get_batch_weight(model, unique_indices)

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()  # (B, L-1, V)
        shift_labels = labels[..., 1:].contiguous()      # (B, L-1)
        shift_labels = shift_labels.to(shift_logits.device)

        bsz, max_seq_len, vocab_size = shift_logits.shape
        # weights: (B,) -> expand to (B, L-1), then flatten
        weights = weights.unsqueeze(1).expand(-1, max_seq_len).reshape(-1) * bsz

        shift_probs = torch.softmax(shift_logits.view(-1, vocab_size), dim=-1)
        shift_labels = shift_labels.view(-1)

        # 예: ignore index < -10
        valid_mask = shift_labels > -10
        shift_labels = shift_labels * valid_mask

        loss = torch.sum(
            -torch.log(
                shift_probs[range(shift_labels.shape[0]), shift_labels]
            ) * valid_mask * weights
        ) / torch.sum(valid_mask)
        return loss
        
    def _get_weighted_loss_t(self, model, unique_indices, logits, labels):

        if hasattr(model, "module"): 
            model = model.module


        weights_batch = self._get_batch_weight(model, unique_indices)


        # Shift so that tokens < n predict n
        # logits: (B, L, V)
        # labels: (B, L)
        shift_logits = logits[..., :-1, :].contiguous()  # => (B, L-1, V)
        shift_labels = labels[..., 1:].contiguous()      # => (B, L-1)
        shift_labels = shift_labels.to(shift_logits.device)

        B, seq_len, vocab_size = shift_logits.shape  # B, L-1, V
        shift_logits = shift_logits.view(-1, vocab_size)     # => ((B * (L-1)), V)
        shift_labels = shift_labels.view(-1)                 # => (B * (L-1))


        valid_mask = (shift_labels != -100)  


        shift_logits = shift_logits[valid_mask]  # (num_valid, V)
        shift_labels = shift_labels[valid_mask]  # (num_valid,)

 
        shift_probs = torch.softmax(shift_logits, dim=-1)  # (num_valid, V)

        weights_for_positions = weights_batch.unsqueeze(1).expand(-1, seq_len)  
        weights_for_positions = weights_for_positions.reshape(-1)  
 
        weights_valid = weights_for_positions[valid_mask] 

        log_probs = -torch.log(
            shift_probs[torch.arange(shift_labels.size(0)), shift_labels]
        )  # (num_valid,)
        weighted_loss = log_probs * weights_valid  # (num_valid,)

        loss = weighted_loss.mean()

        return loss

    def _get_batch_weight(self, model, unique_indices):
        try:
            actual_model = model.module
        except AttributeError:
            actual_model = model


        if self.feature_extractor_setting == 'clip':
            dtype = model.get_score_net_dtype()
            scores = torch.stack(
                [self.clip_feat[str(idx)] for idx in unique_indices], dim=0
            ).to(self.args.device, dtype=dtype)

        elif self.feature_extractor_setting == 'scores':
            scores = [
                [sc[str(idx)] for sc in self.score_dicts]
                for idx in unique_indices
            ]
            scores = torch.tensor(scores).to(self.args.device).bfloat16()

        else:
            raise NotImplementedError

        # model.predict_weights(scores) -> (B,1)
        weights = torch.softmax(model.predict_weights(scores), dim=0)
        # shape (B, 1) => (B,)
        return weights.squeeze(-1)
