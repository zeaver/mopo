# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
from collections import defaultdict, OrderedDict
from typing import List, Dict, Tuple
import numpy as np
import numpy.typing as npt
from tqdm import tqdm
import logging
from pathlib import Path
import json, time

import torch
import torch.distributed as dist
# import torch.nn.Module as TorchModule

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch

from beir.reranking.models import CrossEncoder
from beir.reranking import Rerank

import src.dist_utils as dist_utils
import src.normalize_text as normalize_text
from src.utils import load_pickle, read_jsonl, calculate_time
# from src.summarizer_modeling import Summarizer

logger = logging.getLogger(__name__)

class DenseEncoderModel:
    def __init__(
        self,
        query_encoder,
        doc_encoder=None,
        tokenizer=None,
        max_length=512,
        add_special_tokens=True,
        norm_query=False,
        norm_doc=False,
        lower_case=False,
        normalize_text=False,
        add_prefix=True,
        query_prefix="search_query",
        doc_prefix="search_document",
        **kwargs,
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
        self.lower_case = lower_case
        self.normalize_text = normalize_text
        self.add_prefix=add_prefix
        self.query_prefix = query_prefix
        self.doc_prefix = doc_prefix

    @calculate_time
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized():
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))

        queries = [queries[i] for i in idx]
        if self.normalize_text:
            queries = [normalize_text.normalize(q) for q in queries]
        if self.lower_case:
            queries = [q.lower() for q in queries]

        queries = [self._add_text_prefix(self.query_prefix, q) for q in queries]

        _d = "\n".join(queries[-3:-1])
        logger.info(f"Query demo:\n{_d}")
        
        allemb = []
        nbatch = (len(queries) - 1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                qencode = {key: value.cuda() for key, value in qencode.items()}
                emb = self.query_encoder(**qencode, normalize=self.norm_query)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        print(f"queries embedding shape: {allemb.shape}")
        return allemb

    @staticmethod
    def get_para_string(para):
        if len(para["title"]) > 0:
            return para["title"].strip() + ". " + para["text"]
        else:
            return para["text"]

    def _add_text_prefix(self, prefix, text):
        if self.add_prefix and prefix is not None:
            return f"{prefix}: {text}"
        else:
            return text

    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized():
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]
        corpus = [self.get_para_string(c) for c in corpus]
        if self.normalize_text:
            corpus = [normalize_text.normalize(c) for c in corpus]
        if self.lower_case:
            corpus = [c.lower() for c in corpus]

        corpus = [self._add_text_prefix(self.doc_prefix, c) for c in corpus]

        _d = "\n".join(corpus[-3:-1])
        logger.info(f"Corpus demo:\n{_d}")

        allemb = []
        nbatch = (len(corpus) - 1) // batch_size + 1
        with torch.no_grad():
            for k in tqdm(range(nbatch), desc="Encoding corpus:"):
                start_idx = k * batch_size
                end_idx = min((k + 1) * batch_size, len(corpus))

                cencode = self.tokenizer.batch_encode_plus(
                    corpus[start_idx:end_idx],
                    max_length=self.max_length,
                    padding=True,
                    truncation=True,
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt",
                )
                cencode = {key: value.cuda() for key, value in cencode.items()}
                emb = self.doc_encoder(**cencode, normalize=self.norm_doc)
                allemb.append(emb.cpu())

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        print(f"corpus embedding shape: {allemb.shape}")
        return allemb


class MdrBeirDataloader(GenericDataLoader):

    def __init__(
        self,
        data_folder: str = None,
        prefix: str = None,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "queries.jsonl",
        qrels_folder: str = "qrels",
        qrels_file: str = "",
        test_raw_file: str = "test_info.jsonl",
    ):
        super().__init__(data_folder, prefix, corpus_file, query_file, qrels_folder, qrels_file)
        self.test_raw_file = (os.path.join(data_folder, test_raw_file) if data_folder else test_raw_file)

    def load(self, 
             split: str = "test", 
             load_raw_labels: bool = False):

        corpus, queries, qrels = super().load(split)
        if load_raw_labels:
            # load raw test file info
            self.check(fIn=self.test_raw_file, ext="jsonl")
            with open(self.test_raw_file, encoding='utf8') as fIn:
                res = [json.loads(line.strip()) for line in fIn.readlines()]
            self.test_raw = {x["_id"]:x for x in res}

            return corpus, queries, qrels, self.test_raw
        else:
            return corpus, queries, qrels
        
    def reload_queries(self,
                       split: str = "test"):
            
        logger.info("Re-loading Queries...")
        self.queries = {}
        self.qrels = {}
        self._load_queries()
        self.qrels_file = os.path.join(self.qrels_folder, split + ".tsv")
        if os.path.exists(self.qrels_file):
            self._load_qrels()
            self.queries = {qid: self.queries[qid] for qid in self.qrels}
            logger.info("Loaded %d %s Queries.", len(self.queries), split.upper())
            logger.info("Query Example: %s", list(self.queries.values())[0])
        
        return self.queries, self.qrels
    
    def eval_recall(self,
                    topks,
                    retrieval_results,
                    labels):
        
        recall_res = {f"Recall@{k}":[] for k in topks}

        for qid, qid_docs in retrieval_results.items():
            sorted_docs = sorted(qid_docs, key=qid_docs.get, reverse=True)
            for k in topks:
                recall_k = 0
                for label in labels[qid].keys():
                    if label in sorted_docs[:k]:
                        recall_k =+ 0.5
                recall_res[f"Recall@{k}"].append(recall_k)
        recall_res = {k:np.around(np.mean(v)*100, 2) for k, v in recall_res.items()}
        return recall_res


class MdrFaissSearch(FlatIPFaissSearch):

    def __init__(
        self,
        model,
        batch_size: int = 128,
        corpus_chunk_size: int = 50000,
        use_gpu: bool = False,
        **kwargs,
    ):
        if kwargs:
            _tmp_kws = kwargs.update({"use_gpu":use_gpu})
        else:
            _tmp_kws = {"use_gpu":use_gpu}
        super().__init__(model, batch_size, corpus_chunk_size, **_tmp_kws)

    def hop_search(self, 
        corpus: Dict[str, Dict[str, str]],
        queries: Dict[str, str], 
        topk: int=100,
        score_function:str = "dot", 
        **kwargs
        ) -> Tuple[Dict[str, Dict[str, float]], npt.NDArray, npt.NDArray]: # type: ignore

        assert score_function in self.score_functions
        normalize_embeddings = True if score_function == "cos_sim" else False

        if not self.faiss_index: self.index(corpus, score_function)

        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in queries]
        logger.info("Computing Query Embeddings. Normalize: {}...".format(normalize_embeddings))

        # 1.1 encoding
        query_embeddings = self.model.encode_queries(
            queries, show_progress_bar=True, 
            batch_size=self.batch_size, 
            normalize_embeddings=normalize_embeddings)

        # 1.2 search and collect results
        faiss_scores, faiss_doc_ids = self.faiss_index.search(query_embeddings, topk, **kwargs)
        # total_doc_ids = []
        search_results = {}
        for idx in range(len(query_ids)):
            scores = [float(score) for score in faiss_scores[idx]]
            if len(self.rev_mapping) != 0:
                doc_ids = [self.rev_mapping[doc_id] for doc_id in faiss_doc_ids[idx]]
            else:
                doc_ids = [str(doc_id) for doc_id in faiss_doc_ids[idx]]
            # total_doc_ids.append(doc_ids)
            search_results[query_ids[idx]] = OrderedDict(zip(doc_ids, scores))

        return search_results, faiss_scores, faiss_doc_ids

    @staticmethod
    def _load_results_and_scores(
        data_path, res_name: str = "res.pkl", score_name: str = "scores.pkl"
    ):
        _res = load_pickle(Path(str(data_path)) / res_name)
        _scores = load_pickle(Path(str(data_path)) / score_name)
        return _res, _scores

    @staticmethod
    def _load_hop2_queries(data_path, model_name="t5"):
        raw_hop2 = read_jsonl(Path(str(data_path)) / f"{model_name}_hop2_outputs.jsonl")
        return raw_hop2

    def _resort_hop2(self, hop2_queries, raw_queries, beam_num):
        converted_hop2_queries = {}
        for hop2_info in hop2_queries:
            _idx_info = hop2_info.get("_id").split("_beam")
            assert len(_idx_info) == 2
            raw_id, beam_i = _idx_info[0], _idx_info[1]
            _tmp = {
                "doc_id": hop2_info.get("hop2_info"),
                "hop2_query": self.cat(raw_queries[raw_id], hop2_info.get("hop2_generated_query"))
            }
            if raw_id not in converted_hop2_queries:
                converted_hop2_queries[raw_id] = {beam_i: _tmp}
            else:
                converted_hop2_queries[raw_id][beam_i] = _tmp

        query_ids = list(raw_queries.keys())
        hop2_queries_dicts = OrderedDict()
        for qid in query_ids:
            for nn in range(beam_num):
                hop2_queries_dicts[f"{qid}_beam{nn}"] = converted_hop2_queries[qid][str(nn)].get("hop2_query")
        assert len(hop2_queries_dicts.keys()) == len(hop2_queries)
        return hop2_queries_dicts, converted_hop2_queries

    @staticmethod
    def cat(q, next_q):
        return f"{q} Already know: {next_q}"
