import torch
import random
import json
import sys
import numpy as np
from src import normalize_text


class HotpotCorpusDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        training=False,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
        **kwargs
    ):
        self.training = training
        self.normalize_fn = normalize_text.normalize if normalize else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)
        print(f"Loaded {len(self.data)} examples in {global_rank} GPU")

    def __len__(self):
        return len(self.data)
    
    @staticmethod
    def get_para_string(para):
        if len(para["title"]) > 0:
            return para["title"].strip() + ". " + para["text"].strip()
        else:
            return para["text"].strip()

    def __getitem__(self, index):
        example = self.data[index]

        my_example = {
            "query": self.normalize_fn(self.get_para_string(example)),
            "doc_id": example["_id"],
            "abs_idx": example["abs_idx"],
        }
        return my_example

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = -1
        self.data = []
        path=datapaths
        path = str(path)
        if path.endswith(".jsonl"):
            file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
        elif path.endswith(".json"):
            file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
        self.data.extend(file_data)

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            example["abs_idx"] = counter
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            for line in fin.readlines():
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                example["abs_idx"] = counter
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter



class HotpotCollator(object):
    def __init__(self, tokenizer, passage_maxlength=512):
        self.tokenizer = tokenizer
        self.passage_maxlength = passage_maxlength

    def __call__(self, batch):
        queries = [ex["query"] for ex in batch]
        doc_ids = [ex["doc_id"] for ex in batch]
        abs_idx = [int(ex["abs_idx"]) for ex in batch]

        qout = self.tokenizer.batch_encode_plus(
            queries,
            max_length=self.passage_maxlength,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        q_tokens, q_mask = qout["input_ids"], qout["attention_mask"].bool()

        batch = {
            "input_ids": q_tokens,
            "attention_mask": q_mask,
            "doc_ids": doc_ids,
            "abs_idx": torch.tensor(abs_idx),
        }

        return batch