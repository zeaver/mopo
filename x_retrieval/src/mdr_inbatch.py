# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
from pathlib import Path
import copy
import logging

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

import transformers

from src import contriever, dist_utils, utils

logger = logging.getLogger(__name__)


class MdrInBatch(nn.Module):
    def __init__(self, opt, retriever, tokenizer):
        super(MdrInBatch, self).__init__()

        self.opt = opt
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        self.tokenizer = tokenizer
        self.encoder = retriever
        
        self.gck_segment = opt.gck_segment
        if self.gck_segment <= 1 or self.gck_segment == opt.per_gpu_batch_size:
            self.use_gck = False
        else:
            self.use_gck = True
        
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
        self.info_nce_weight = 1

        if opt.if_load_teacher_retriever:
            self.teacher_optimization_mode = opt.teacher_optimization_mode
            if self.teacher_optimization_mode == "vanilla" and opt.teacher_retriever_id is not None:
                assert Path(str(opt.teacher_retriever_id)).exists()
                self.teacher, _, _ = contriever.load_retriever(opt.model_name, opt.teacher_retriever_id)
                for name, parameter in self.teacher.named_parameters():
                    parameter.requires_grad = False

            elif self.teacher_optimization_mode == "moco":
                self.momentum = opt.momentum
                self.teacher = copy.deepcopy(self.encoder)
                for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
                    param_teacher.data.copy_(param_encoder.data)
                    param_teacher.requires_grad = False

            self.log_softmax = nn.LogSoftmax(dim=-1)
            self.teacher_kl_loss = nn.KLDivLoss(reduction="batchmean", log_target=True)
            assert isinstance(self.opt.teacher_loss_weight, float) and self.opt.teacher_loss_weight >= 0
            self.teacher_loss_weight = self.opt.teacher_loss_weight
            if opt.teacher_abaltion:
                self.info_nce_weight = 0
                logger.warning(f"InfoNCE loss weight is set to \"0\", place check whether doing ablation study")
        else:
            self.teacher = None

    def get_encoder(self):
        return self.encoder
    
    def get_embedding(self, emb_model, input_ids, attention_mask, normalize):
        if not self.use_gck:
            return emb_model(input_ids=input_ids, attention_mask=attention_mask, normalize=normalize)
        else:
            pooled_output = []
            for mini_batch in range(0, input_ids.shape[0], self.gck_segment):
                gck_input_dict = {
                    "input_ids": input_ids[mini_batch:mini_batch + self.gck_segment],
                    "attention_mask": attention_mask[mini_batch:mini_batch + self.gck_segment],
                    "normalize": normalize,
                }
                mini_batch_pooled_output = checkpoint(emb_model, use_reentrant=False, **gck_input_dict)
                pooled_output.append(mini_batch_pooled_output)
            return torch.cat(pooled_output, dim=0)

    def forward(self, q1_tokens, q1_mask, q2_tokens, q2_mask, k_tokens, k_mask, stats_prefix="", iter_stats={}, **kwargs):
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        bsz = len(q1_tokens)

        # TODO: refactor to _model_forward
        q1_emb = self.get_embedding(emb_model=self.encoder, input_ids=q1_tokens, attention_mask=q1_mask, normalize=self.norm_query)
        q2_emb = self.get_embedding(emb_model=self.encoder, input_ids=q2_tokens, attention_mask=q2_mask, normalize=self.norm_query)
        kemb = self.get_embedding(emb_model=self.encoder, input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)
        dim_shift = dist_utils.get_rank() * len(kemb)

        hop1_score, hop1_target = self.compute_qk_scores(q1_emb, gather_kemb, dim_shift + bsz, dim_shift)
        hop2_score, hop2_target = self.compute_qk_scores(q2_emb, gather_kemb, dim_shift, dim_shift + bsz)
        # scores = torch.cat([hop1_score, hop2_score], dim=0)
        # labels = torch.cat([hop1_target, hop2_target], dim=0)
        hop1_loss = self.loss_fct(hop1_score, hop1_target)
        hop2_loss = self.loss_fct(hop2_score, hop2_target)
        loss = hop1_loss + hop2_loss

        if self.teacher:
            if self.teacher_optimization_mode == "moco":
                with torch.no_grad():
                    self._momentum_update_key_encoder()
            teacher_args = self._prepare_teacher_input_args(kwargs, k_tokens=k_tokens, k_mask=k_mask)
            teacher_hop1_scores, teacher_hop2_scores, _, _ = self.teacher_forward(**teacher_args)
            log_softmax_hop1_scores = self.log_softmax(hop1_score)
            log_softmax_hop2_scores = self.log_softmax(hop2_score)
            hop1_kl_loss = self.teacher_kl_loss(log_softmax_hop1_scores, self.log_softmax(teacher_hop1_scores))
            hop2_kl_loss = self.teacher_kl_loss(log_softmax_hop2_scores, self.log_softmax(teacher_hop2_scores))
            kl_loss = hop1_kl_loss + hop2_kl_loss
            
            loss = kl_loss * self.teacher_loss_weight + loss * self.info_nce_weight

            iter_stats[f"{stats_prefix}total loss"] = (loss.item(), bsz)

            iter_stats[f"{stats_prefix}stu loss"] = (hop1_loss.item() + hop2_loss.item(), bsz)
            iter_stats[f"{stats_prefix}KL loss"] = (self.teacher_loss_weight * kl_loss.item(), bsz)
            
            iter_stats[f"{stats_prefix}base hop1 loss"] = (hop1_loss.item(), bsz)
            iter_stats[f"{stats_prefix}base hop2 loss"] = (hop2_loss.item(), bsz)
            iter_stats[f"{stats_prefix}KL hop1 loss"] = (hop1_kl_loss.item(), bsz)
            iter_stats[f"{stats_prefix}KL hop2 loss"] = (hop2_kl_loss.item(), bsz)
        else:
            iter_stats[f"{stats_prefix}total loss"] = (loss.item(), bsz)
            iter_stats[f"{stats_prefix}base hop1 loss"] = (hop1_loss.item(), bsz)
            iter_stats[f"{stats_prefix}base hop2 loss"] = (hop2_loss.item(), bsz)

        hop1_predicted_idx = torch.argmax(hop1_score, dim=-1)
        hop1_accuracy = 100 * (hop1_predicted_idx == hop1_target).float().mean()
        hop2_predicted_idx = torch.argmax(hop2_score, dim=-1)
        hop2_accuracy = 100 * (hop2_predicted_idx == hop2_target).float().mean()

        # stdq = torch.std(torch.cat([q1_emb, q2_emb], dim=0), dim=0).mean().item()
        # stdk = torch.std(kemb, dim=0).mean().item()
        iter_stats[f"{stats_prefix}hop1 accuracy"] = (hop1_accuracy, bsz)
        iter_stats[f"{stats_prefix}hop2 accuracy"] = (hop2_accuracy, bsz)
        # iter_stats[f"{stats_prefix}stdq"] = (stdq, bsz)
        # iter_stats[f"{stats_prefix}stdk"] = (stdk, bsz)

        return loss, iter_stats
    

    def compute_qk_scores(self, q_embeddings, k_embeddings, mask_shift, label_shift):
        bsz = q_embeddings.size(0)
        score = torch.einsum("id, jd->ij", q_embeddings / self.opt.temperature, k_embeddings)
        score_mask =  torch.zeros_like(score).to(q_embeddings.device)
        score_mask[:, mask_shift : mask_shift + bsz] = torch.eye(bsz)
        score = score.float().masked_fill(score_mask.bool(), -1e9).type_as(score)
        target = torch.arange(bsz, dtype=torch.long, device=q_embeddings.device) + label_shift

        return score, target
    
    def teacher_forward(self, s1_tokens, s1_mask, s2_tokens, s2_mask, k_tokens, k_mask):
        bsz = s1_tokens.size(0)
        teacher_s1_emb = self.get_embedding(emb_model=self.teacher, input_ids=s1_tokens, attention_mask=s1_mask, normalize=self.norm_query)
        teacher_s2_emb = self.get_embedding(emb_model=self.teacher,input_ids=s2_tokens, attention_mask=s2_mask, normalize=self.norm_query)
        teacher_k_emb = self.get_embedding(emb_model=self.teacher,input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)
        gather_fn = dist_utils.gather
        gather_teacher_k_emb = gather_fn(teacher_k_emb)
        dim_shift = dist_utils.get_rank() * len(teacher_k_emb)

        teacher_hop1_score, teacher_hop1_target = self.compute_qk_scores(teacher_s1_emb, gather_teacher_k_emb, dim_shift + bsz, dim_shift)
        teacher_hop2_score, teacher_hop2_target = self.compute_qk_scores(teacher_s2_emb, gather_teacher_k_emb, dim_shift, dim_shift + bsz)
        # _teacher_scores = torch.cat([teacher_hop1_score, teacher_hop2_score], dim=0)
        # _teacher_labels = torch.cat([teacher_hop1_target, teacher_hop2_target], dim=0)
        # return _teacher_scores, _teacher_labels
        return (
            teacher_hop1_score, 
            teacher_hop2_score,
            teacher_hop1_target,
            teacher_hop2_target,
        )
    
    @staticmethod
    def _prepare_teacher_input_args(input_dicts, k_tokens, k_mask):
        return {
            "s1_tokens": input_dicts["s1_tokens"],
            "s1_mask": input_dicts["s1_mask"],
            "s2_tokens": input_dicts["s2_tokens"],
            "s2_mask": input_dicts["s2_mask"],
            "k_tokens": k_tokens,
            "k_mask": k_mask,
        }
    
    def _momentum_update_key_encoder(self):
        """
        Update of the key encoder
        """
        for param_encoder, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
            param_teacher.data = param_teacher.data * self.momentum + param_encoder.data * (1.0 - self.momentum)
    
    @torch.no_grad()
    def batch_eval(self, q1_tokens, q1_mask, q2_tokens, q2_mask, k_tokens, k_mask, **kwargs):
        bsz = len(q1_tokens)

        q1_emb = self.get_embedding(emb_model=self.encoder, input_ids=q1_tokens, attention_mask=q1_mask, normalize=self.norm_query)
        q2_emb = self.get_embedding(emb_model=self.encoder, input_ids=q2_tokens, attention_mask=q2_mask, normalize=self.norm_query)
        kemb = self.get_embedding(emb_model=self.encoder, input_ids=k_tokens, attention_mask=k_mask, normalize=self.norm_doc)

        hop1_score, hop1_target = self.compute_qk_scores(q1_emb, kemb, bsz, 0)
        hop2_score, hop2_target = self.compute_qk_scores(q2_emb, kemb, 0, bsz)

        ranked_1_hop = hop1_score.argsort(dim=1, descending=True)
        ranked_2_hop = hop2_score.argsort(dim=1, descending=True)
        idx2ranked_1 = ranked_1_hop.argsort(dim=1)
        idx2ranked_2 = ranked_2_hop.argsort(dim=1)
        rrs_1, rrs_2 = [], []
        for t, idx2ranked in zip(hop1_target, idx2ranked_1):
            rrs_1.append(1 / (idx2ranked[t].item() + 1))
        for t, idx2ranked in zip(hop2_target, idx2ranked_2):
            rrs_2.append(1 / (idx2ranked[t].item() + 1))

        hop_1_predicted_idx = torch.argmax(hop1_score, dim=-1)
        hop_2_predicted_idx = torch.argmax(hop2_score, dim=-1)
        
        hop_1_acc = (hop_1_predicted_idx == hop1_target).float()
        hop_2_acc = (hop_2_predicted_idx == hop2_target).float()
        return {"rrs_1": rrs_1, "rrs_2": rrs_2, "acc_1": hop_1_acc, "acc_2": hop_2_acc}


