# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import math
import random
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

import transformers

from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)


class InBatch(nn.Module):
    def __init__(self, opt, retriever=None, tokenizer=None):
        super(InBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        if retriever is None or tokenizer is None:
            retriever, tokenizer = self._load_retriever(
                opt.retriever_model_id, pooling=opt.pooling, random_init=opt.random_init
            )
        self.tokenizer = tokenizer
        self.encoder = retriever
        
        self.gck_segment = opt.gck_segment
        if self.gck_segment <= 1 or self.gck_segment == opt.per_gpu_batch_size:
            self.use_gck = False
        else:
            self.use_gck = True

    def _load_retriever(self, model_id, pooling, random_init):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if "xlm" in model_id:
            model_class = contriever.XLMRetriever
        else:
            model_class = contriever.Contriever

        if random_init:
            retriever = model_class(cfg)
        else:
            retriever = utils.load_hf(model_class, model_id)

        if "bert-" in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.config.pooling = pooling

        return retriever, tokenizer

    def get_encoder(self):
        return self.encoder
    
    def get_embedding(self, input_ids, attention_mask, normalize):
        if not self.use_gck:
            return self.encoder(input_ids=input_ids, attention_mask=attention_mask, normalize=normalize)
        else:
            pooled_output = []
            for mini_batch in range(0, input_ids.shape[0], self.gck_segment):
                gck_input_dict = {
                    "input_ids": input_ids[mini_batch:mini_batch + self.gck_segment],
                    "attention_mask": attention_mask[mini_batch:mini_batch + self.gck_segment],
                    "normalize": normalize,
                }
                mini_batch_pooled_output = checkpoint(self.encoder, use_reentrant=False, **gck_input_dict)
                pooled_output.append(mini_batch_pooled_output)
            return torch.cat(pooled_output, dim=0)

    def forward(self, q_tokens, q_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):

        bsz = len(q_tokens)
        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)

        qemb = self.get_embedding(input_ids=q_tokens, attention_mask=q_mask, normalize=self.norm_query)
        kemb = self.get_embedding(input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather

        gather_kemb = gather_fn(kemb)

        labels = labels + dist_utils.get_rank() * len(kemb)

        scores = torch.einsum("id, jd->ij", qemb / self.opt.temperature, gather_kemb)

        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # if dist_utils.is_main():
        #     print("=======================================================")
        #     print("=======================================================")
        #     print("scores size:", scores.size())
        #     print("q_tokens size:", q_tokens.size())
        #     print("k_tokens size:", k_tokens.size())

        # log stats
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = (loss.item(), bsz)

        predicted_idx = torch.argmax(scores, dim=-1)
        accuracy = 100 * (predicted_idx == labels).float().mean()
        stdq = torch.std(qemb, dim=0).mean().item()
        stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}accuracy"] = (accuracy, bsz)
        iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats
