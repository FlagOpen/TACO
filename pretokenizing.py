import os
import json
import shutil
import logging
import multiprocessing
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple

import datasets
import transformers
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer, HfArgumentParser

from datamodule import DATASET_NAME


@dataclass
class PretokenizationArguments:
    """
    Configuration for data pretokenization.
    """
    tokenizer_dir: Optional[str] = field(
        default="bigcode/starcoder", metadata={"help": "Name or path to the tokenizer."}
    )
    data_path: Optional[str] = field(
        default=None, metadata={"help": "Path to the local training dataset."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Dir to the dataset to pretokenize."}
    )
    dataset_name: Optional[str] = field(
        default="starcoder_tokenized", metadata={"help": "Name or path to the dataset to pretokenize."}
    )
    num_workers: Optional[int] = field(default=None, metadata={"help": "Number of workers used for code evaluation."})
    

def initialize(
    dataset: datasets.Dataset,
    tokenizer: transformers.PreTrainedTokenizer
) -> Tuple[Sequence[str], Sequence[str]]:
    
    sources = []
    targets = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        try:
            solutions = json.loads(sample["solutions"])
        except ValueError:
            continue
        starter_code = None if len(sample["starter_code"]) == 0 else sample["starter_code"]
        try:
            input_outpout = json.loads(sample["input_output"])
            fn_name = (
                None if not input_outpout.get("fn_name") else input_outpout["fn_name"]
            )
        except ValueError:
            fn_name = None
        prompt = "\nQUESTION:\n"
        prompt += sample["question"]
        if starter_code:
            prompt += starter_code
        if (not fn_name) and (not starter_code):
            call_format = "\nUse Standard Input format"
            prompt += call_format
        else:
            call_format = "\nUse Call-Based format"
            prompt += call_format
        prompt += "\nANSWER:\n"
        for solution_str in solutions:
            sources.append(prompt)
            targets.append(f"{solution_str}{tokenizer.eos_token}")    
    return datasets.Dataset.from_dict({"source": sources, "target": targets})   


def tokenize_function(example):
    combined_text = example["source"] + example['target']
    source_text = example["source"]
    input_ids = tokenizer(combined_text, truncation=True, max_length=2048)["input_ids"]
    source_ids_lens = len(tokenizer(source_text, truncation=True, max_length=2048)["input_ids"])

    return {
        "input_ids": input_ids,
        "source_ids_lens": source_ids_lens,
    }


if __name__ == '__main__':
    
    parser = HfArgumentParser(PretokenizationArguments)
    args = parser.parse_args()
    if args.num_workers is None:
        args.num_workers = multiprocessing.cpu_count()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_auth_token=True)

    logging.warning("Loading data...")
    if os.path.exists(args.data_path):
        raw_dataset = load_from_disk(args.data_path)['train']
    else:
        raw_dataset = load_dataset(DATASET_NAME, split="train")
    
    logging.warning("Formatting inputs...")
    arrow_path = os.path.join(args.cache_dir, 'tmp_dataset')
    initialized_dataset = initialize(raw_dataset, tokenizer)

    # for big tables, we have to write them on disk: 
    # https://github.com/huggingface/datasets/pull/2150
    initialized_dataset.save_to_disk(arrow_path)
    # reload from Arrow format
    initialized_dataset = load_from_disk(arrow_path)
    
    logging.warning("Tokenizing inputs... This may take some time...")
    tokenized_dataset = initialized_dataset.map(
                            tokenize_function, 
                            num_proc=args.num_workers,
                            remove_columns=["source", "target"],
                        )
    tokenized_dataset.save_to_disk(os.path.join(args.cache_dir, args.dataset_name))
    shutil.rmtree(arrow_path)