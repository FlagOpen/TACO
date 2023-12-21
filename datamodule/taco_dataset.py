import os
import copy
import logging
from dataclasses import dataclass
from typing import Dict, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from datasets import load_from_disk

from .constants import IGNORE_INDEX


def preprocess(
    input_ids: Sequence[Sequence[int]],
    source_ids_lens: Sequence[int],
) -> Dict:
    """Preprocess the tokenized data"""
    
    input_ids = [torch.tensor(ids) for ids in input_ids]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, source_ids_lens):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def preprocess_scores(
    scores: Sequence[Sequence[float]],
    source_ids_lens: Sequence[int],
    learning_skill: int
) -> Sequence:
    """Preprocess the scores"""
    scores = [torch.tensor(score[:, learning_skill]) for score in scores]
    for score, source_len in zip(scores, source_ids_lens):
        score = torch.nn.functional.pad(score, (source_len, 0), 'constant', 0.0)
    return scores


class TacoDataset(Dataset):
    """Dataset for fine-tune."""
    
    def __init__(self, data_path: str, debug: bool=False, learning_skill: int=None):
        super(TacoDataset, self).__init__()
        logging.warning("Loading tokenized data...")
        if os.path.exists(data_path):
            dataset = load_from_disk(data_path).shuffle()
        else:
            raise ValueError(" The specified data_path does not exist. Please provide a tokenized dataset")
        
        if not all(key in dataset.column_names for key in ['input_ids', 'source_ids_lens']):
            raise ValueError("Data has not been tokenized. Please tokenize the data first.")
        if debug:
            dataset = dataset.select(range(1000))
        if learning_skill:
            dataset = dataset.filter(lambda entry: entry['labels'][learning_skill])
        
        logging.warning("Collect columns of hf dataset... This may take some time...")
        input_ids = dataset['input_ids']
        source_ids_lens = dataset['source_ids_lens']
        
        self.learning_skill = None
        if learning_skill:
            scores = dataset['scores']
            scores = preprocess_scores(scores, source_ids_lens, learning_skill)
            self.scores = scores
            self.learning_skill = learning_skill
        
        logging.warning("Processing inputs...")
        data_dict = preprocess(input_ids, source_ids_lens)
        
        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.learning_skill:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i], scores=self.scores[i])
        else:
            return dict(input_ids=self.input_ids[i], labels=self.labels[i])
    

@dataclass
class DataCollatorForTacoDataset(object):
    """Collate examples for fine-tune."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
        )
        

@dataclass
class DataCollatorForTacoSkillDataset(object):
    """Collate examples for TACO fine-tune."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels", "scores"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        scores = torch.nn.utils.rnn.pad_sequence(scores, batch_first=True, padding_value=0.0)
        return dict(
            input_ids=input_ids,
            labels=labels,
            scores=scores,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
