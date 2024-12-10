# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Dict, Sequence
import torch


@dataclass
class DataCollatorForSFT(object):
    """Collate examples for SFT
    TODO: attention mask
    """

    pad_token_id: int = 0
    ignore_index: int = -100
    max_length: int = 2048

    def __call__(self, instances: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([
            instance[key][: self.max_length]
            for instance in instances]
            for key in ["input_ids", "labels"]
        )
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=self.ignore_index
        )
        return dict(
            input_ids=input_ids,
            labels=labels,
        )
