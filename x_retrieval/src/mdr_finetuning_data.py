# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import random
import json
import sys
import numpy as np
from src import normalize_text


class MdrDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        datapaths,
        global_rank=-1,
        world_size=-1,
        maxload=None,
        normalize=False,
        data_format="teacher",
        if_add_prefix=False,
        doc_prefix="search_document",
        query_prefix="search_query",
        statement_prefix="search_statement",
    ):
        self.data_format =data_format
        self.normalize_fn = normalize_text.normalize if normalize else lambda x: x
        self._load_data(datapaths, global_rank, world_size, maxload)

        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.statement_prefix = statement_prefix
        self.if_add_prefix = if_add_prefix

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        example = self.data[index]
        if self.data_format == "baseline":
            first_hop_query = self._add_text_prefix(self.query_prefix, example.get("first_hop_query"), self.if_add_prefix)
            first_hop_doc = self._get_para_string(example["first_hop_doc"], add_prefix=self.if_add_prefix)

            _first_doc_text = self._get_para_string(example["first_hop_doc"], add_prefix=False)
            second_hop_query = self.combine_query_and_sub_statement(example.get("first_hop_query"), _first_doc_text, add_prefix=self.if_add_prefix)
            second_hop_doc = self._get_para_string(example["second_hop_doc"], add_prefix=self.if_add_prefix)

            neg_doc1 = self._get_para_string(example["neg_paras"][0], add_prefix=self.if_add_prefix)
            neg_doc2 = self._get_para_string(example["neg_paras"][1], add_prefix=self.if_add_prefix)

            res = {
                "first_hop_query": first_hop_query,
                "first_hop_doc": first_hop_doc,
                "second_hop_query": second_hop_query,
                "second_hop_doc": second_hop_doc,
                "negatives": [neg_doc1, neg_doc2]
            }
        elif self.data_format == "teacher":
            first_hop_query = self._add_text_prefix(self.statement_prefix, example.get("first_hop_statement"), self.if_add_prefix)
            first_hop_doc = self._get_para_string(example["first_hop_doc"], add_prefix=self.if_add_prefix)
            second_hop_query = self._add_text_prefix(self.statement_prefix, example.get("second_hop_statement"), self.if_add_prefix)
            second_hop_doc = self._get_para_string(example["second_hop_doc"], add_prefix=self.if_add_prefix)
            neg_doc1 = self._get_para_string(example["neg_paras"][0])
            neg_doc2 = self._get_para_string(example["neg_paras"][1])
            res = {
                "first_hop_query": first_hop_query,
                "first_hop_doc": first_hop_doc,
                "second_hop_query": second_hop_query,
                "second_hop_doc": second_hop_doc,
                "negatives": [neg_doc1, neg_doc2]
            }
        elif self.data_format == "mix":
            first_hop_query = self._add_text_prefix(self.query_prefix, example.get("first_hop_query"), self.if_add_prefix)
            first_hop_doc = self._get_para_string(example["first_hop_doc"], add_prefix=self.if_add_prefix)

            second_hop_query = self.combine_query_and_sub_statement(example.get("first_hop_query"), example.get("first_hop_statement"), add_prefix=self.if_add_prefix)
            second_hop_doc = self._get_para_string(example["second_hop_doc"], add_prefix=self.if_add_prefix)

            neg_doc1 = self._get_para_string(example["neg_paras"][0], add_prefix=self.if_add_prefix)
            neg_doc2 = self._get_para_string(example["neg_paras"][1], add_prefix=self.if_add_prefix)

            first_hop_statement = self._add_text_prefix(self.statement_prefix, example.get("first_hop_statement"), self.if_add_prefix)
            second_hop_statement = self._add_text_prefix(self.statement_prefix, example.get("second_hop_statement"), self.if_add_prefix)

            res = {
                "first_hop_query": first_hop_query,
                "first_hop_doc": first_hop_doc,
                "second_hop_query": second_hop_query,
                "second_hop_doc": second_hop_doc,
                "first_hop_statement": first_hop_statement,
                "second_hop_statement": second_hop_statement,
                "negatives": [neg_doc1, neg_doc2]
            }


        else:
            raise ValueError(f"data format error, get: {self.data_format}")

        return res

    def _load_data(self, datapaths, global_rank, world_size, maxload):
        counter = 0
        self.data = []
        for path in datapaths:
            path = str(path)
            if path.endswith(".jsonl"):
                file_data, counter = self._load_data_jsonl(path, global_rank, world_size, counter, maxload)
            elif path.endswith(".json"):
                file_data, counter = self._load_data_json(path, global_rank, world_size, counter, maxload)
            self.data.extend(file_data)
            if maxload is not None and maxload > 0 and counter >= maxload:
                break

    def _load_data_json(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        with open(path, "r") as fin:
            data = json.load(fin)
        for example in data:
            counter += 1
            if global_rank > -1 and not counter % world_size == global_rank:
                continue
            examples.append(example)
            if maxload is not None and maxload > 0 and counter == maxload:
                break

        return examples, counter

    def _load_data_jsonl(self, path, global_rank, world_size, counter, maxload=None):
        examples = []
        print(path)
        with open(path, "r") as fin:
            for line in fin:
                counter += 1
                if global_rank > -1 and not counter % world_size == global_rank:
                    continue
                example = json.loads(line)
                examples.append(example)
                if maxload is not None and maxload > 0 and counter == maxload:
                    break

        return examples, counter

    def _get_para_string(self, para, add_prefix=False):
        if add_prefix:
            return f"{self.doc_prefix}: " +  para["title"].strip() + ". " + para["text"] if len(para["title"]) > 0 else para["text"]
        else:
            return para["title"].strip() + ". " + para["text"] if len(para["title"]) > 0 else para["text"]
    
    def combine_query_and_sub_statement(self, query, sub_statement, add_prefix=False):
        if not add_prefix:
            return f"{query} Already know: {sub_statement}"
        else:
            return f"{self.query_prefix}: {query} Already know: {sub_statement}"

    
    @staticmethod
    def _add_text_prefix(prefix, text, if_add_prefix):
        if if_add_prefix:
            return f"{prefix}: {text}"
        else:
            return text


class Collator(object):
    def __init__(self, tokenizer, collator_args):
        self.tokenizer = tokenizer
        self.passage_maxlength = collator_args.passage_length
        self.query_maxlength = collator_args.query_length
        self.data_format = collator_args.data_format

    def call_tokenizer(self, text, max_seq_len):
        res = self.tokenizer.batch_encode_plus(
            text,
            max_length=max_seq_len,
            truncation=True,
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        return res

    def __call__(self, batch):
        first_hop_query = [ex["first_hop_query"] for ex in batch]
        first_hop_doc = [ex["first_hop_doc"] for ex in batch]
        second_hop_query = [ex["second_hop_query"] for ex in batch]
        second_hop_doc = [ex["second_hop_doc"] for ex in batch]
        negs = [item for ex in batch for item in ex["negatives"]]

        queries = first_hop_query + second_hop_query
        allpassages = first_hop_doc + second_hop_doc + negs

        q1_out = self.call_tokenizer(first_hop_query, self.query_maxlength)
        q2_out = self.call_tokenizer(second_hop_query, self.query_maxlength)
        kout = self.call_tokenizer(allpassages, self.passage_maxlength)
        q1_tokens, q1_mask = q1_out["input_ids"], q1_out["attention_mask"].bool()
        q2_tokens, q2_mask = q2_out["input_ids"], q2_out["attention_mask"].bool()
        k_tokens, k_mask = kout["input_ids"], kout["attention_mask"].bool()

        g_tokens, g_mask = k_tokens[: len(queries)], k_mask[: len(queries)]
        n_tokens, n_mask = k_tokens[len(queries) :], k_mask[len(queries) :]

        res_batch = {
            "q1_tokens": q1_tokens,
            "q1_mask": q1_mask,
            "q2_tokens": q2_tokens,
            "q2_mask": q2_mask,
            "k_tokens": k_tokens,
            "k_mask": k_mask,
            "g_tokens": g_tokens,
            "g_mask": g_mask,
            "n_tokens": n_tokens,
            "n_mask": n_mask,
        }

        if self.data_format == "mix":
            first_hop_statement = [ex["first_hop_statement"] for ex in batch]
            second_hop_statement = [ex["second_hop_statement"] for ex in batch]
            s1 = self.call_tokenizer(first_hop_statement, self.query_maxlength)
            s2 = self.call_tokenizer(second_hop_statement, self.query_maxlength)

            s1_tokens, s1_mask = s1["input_ids"], s1["attention_mask"].bool()
            s2_tokens, s2_mask = s2["input_ids"], s2["attention_mask"].bool()
            _added_info = {
                "s1_tokens": s1_tokens,
                "s1_mask": s1_mask,
                "s2_tokens": s2_tokens,
                "s2_mask": s2_mask,
            }
            res_batch.update(_added_info)
            return res_batch
            
        else:
            return res_batch
