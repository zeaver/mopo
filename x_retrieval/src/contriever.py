# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import functools
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import BertModel, XLMRobertaModel, RobertaModel, AutoConfig, AutoTokenizer

from src import utils

logger = logging.getLogger(__name__)

class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)

        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]

        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class RobertaRetriever(RobertaModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, add_pooling_layer=True)
        self.project = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size), nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        emb = last_hidden[:, 0]
        emb = self.project(emb)

        return emb

class XLMRetriever(XLMRobertaModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb

class E5Retriever(BertModel):
    def __init__(self, config, **kwargs):
        super().__init__(config, add_pooling_layer=False)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
        **kwargs
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden = model_output["last_hidden_state"]

        # average pooling
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0)
        embeddings = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

        # normalize embeddings
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


MODEL_DICTS = {}

def model_loading_register(model_name):
    def decorator(fct):
        @functools.wraps(fct)
        def wrapped_fct(*args, **kwargs):
            return fct(*args, **kwargs)
        MODEL_DICTS[model_name] = wrapped_fct
        return wrapped_fct
    return decorator

@model_loading_register("contriever")
def _load_contriever(model_path,
                     raw_model_id=r"huggingface_download/contriever-msmarco"):
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            # retriever_model_id = "bert-base-uncased"
            retriever_model_id = "bert-base-multilingual-cased"
        if not os.path.exists(retriever_model_id):
            retriever_model_id = raw_model_id
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
        if "xlm" in retriever_model_id:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        retriever = model_class(cfg)
        pretrained_dict = pretrained_dict["model"]
        retriever.load_state_dict(pretrained_dict, strict=True)
    else:
        retriever_model_id = model_path
        if "xlm" in retriever_model_id:
            model_class = XLMRetriever
        else:
            model_class = Contriever
        cfg = utils.load_hf(transformers.AutoConfig, model_path)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
        retriever = utils.load_hf(model_class, model_path)

    return retriever, tokenizer, retriever_model_id

@model_loading_register("facebook_mdr")
def _load_meta_mdr(model_path,
                   raw_model_id="huggingface_download/roberta-base"):
    bert_config = AutoConfig.from_pretrained(raw_model_id)
    tokenizer = AutoTokenizer.from_pretrained(raw_model_id)

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    def filter(x): return x[8:] if x.startswith('encoder.') else x
    
    retriever = RobertaRetriever(bert_config)
    state_dict = {filter(k): v for (k, v) in state_dict.items() if filter(k) in retriever.state_dict()}
    retriever.load_state_dict(state_dict, strict=True)
    return retriever, tokenizer, "roberta-base"

@model_loading_register("e5")
def _load_e5(model_path, 
             raw_model_id=r"download/e5_base"
             ):
    logger.warning(f"e5 seriers need normalization at the end of encoding pipeline, please check config")
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict["opt"]
        if hasattr(opt, "retriever_model_id"):
            retriever_model_id = opt.retriever_model_id
        else:
            retriever_model_id = raw_model_id
        if not os.path.exists(retriever_model_id):
            retriever_model_id = raw_model_id

        tokenizer = utils.load_hf(transformers.AutoTokenizer, raw_model_id)
        cfg = utils.load_hf(transformers.AutoConfig, raw_model_id)
        model_class = E5Retriever
        retriever = model_class(cfg)
        pretrained_dict = pretrained_dict["model"]
        retriever.load_state_dict(pretrained_dict, strict=False)
    else:
        bert_config = AutoConfig.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        retriever = E5Retriever.from_pretrained(model_path, local_files_only=True)

    return retriever, tokenizer, "e5-base-vx"


def load_retriever(base_model_type, model_path, *args, **kwargs):
    try:
        func = MODEL_DICTS[base_model_type]
        return func(model_path, *args, **kwargs)
    except KeyError:
        raise ValueError(f"Invalid key: {base_model_type}. Function not found in our modeling file. Expected: {MODEL_DICTS.keys()}")