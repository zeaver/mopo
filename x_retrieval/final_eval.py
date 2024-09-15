import pdb
import os
import time
import sys
import torch 
from typing import Dict, List
import logging
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict, namedtuple
import argparse


import sys
sys.path.append(r"x_retrieval")

from src.options import Options
from src import beir_utils, dist_utils, contriever, utils

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default='', type=str, help="model path")
parser.add_argument("--hop1_res_path", default='', type=str, help="hop1 res path")
parser.add_argument("--hop2_quries_path", default='', type=str, help="hop2 quries path")
parser.add_argument("--faiss_path", default='', type=str, help="faiss path")


args = parser.parse_args()

data_path = r"x_retrieval/datasets/beir_hotpot"
corpus, queries, qrels, test_raw = beir_utils.MdrBeirDataloader(data_folder=data_path).load("test", True)

model_name = "e5"
model_path = args.model_path
raw_model_id = "download/e5_base_v2"
eval_mode, tokenizer, retriever_model_id = contriever.load_retriever(
    model_name, model_path, raw_model_id
)

dmodel = beir_utils.DenseEncoderModel(
    query_encoder=eval_mode,
    doc_encoder=eval_mode,
    tokenizer=tokenizer,
    add_prefix=True,
    query_prefix="query",
    doc_prefix='passage',
    norm_doc=True,
    norm_query=True,
)

eval_mode.cuda()

faiss_search = beir_utils.MdrFaissSearch(dmodel, batch_size=512, use_gpu=True)


# ATTENTION：Need to change file_name in the directory XXXX_res/score.pkl -> res/scores.pkl
hop1_res_path = args.hop1_res_path
hop1_result, hop1_socres = faiss_search._load_results_and_scores(hop1_res_path, score_name="scores.pkl")


hop2_queries_path = args.hop2_quries_path
hop2_quries_dicts =  faiss_search._load_hop2_queries(hop2_queries_path)

hop2_queries, hop2_info = faiss_search._resort_hop2(hop2_quries_dicts, queries, 50)

faiss_search.load(input_dir=args.faiss_path)


hop1_beam_size = 50
hop2_beam_size = 100

print("begin search\n")
hop2_result, hop2_socres, hop2_doc_local_idx = faiss_search.hop_search(corpus, hop2_queries, 100)
utils.save_pickle(Path(hop1_res_path)/ f"hop2_retrieval_res_large.pkl", hop2_result)
utils.save_pickle(Path(hop1_res_path)/ f"hop2_retrieval_scores_large.pkl", hop2_socres)
utils.save_pickle(Path(hop1_res_path)/ f"hop2_retrieval_local_ids_large.pkl", hop2_doc_local_idx)


hop2_socres_ = hop2_socres.reshape(len(queries), 50, 100)
hop2_socres_ = hop2_socres_[:,:hop1_beam_size,:hop2_beam_size]

hop1_socres_ = hop1_socres[:, :hop1_beam_size]

path_scores = np.expand_dims(hop1_socres_, axis=2) + hop2_socres_


MdrRecoder = namedtuple("MdrRecoder", ["retrieved_corpus_ids", "hop1_corpus_doc_ids", "path_corpus_doc_ids"])

hop1_doc_local_idx = []
for qid in list(queries.keys()):
    _res = []
    for j in hop1_result[qid].keys():
        _res.append(j)
    hop1_doc_local_idx.append(_res)
hop1_doc_local_idx = np.array(hop1_doc_local_idx)

hop2_doc_local_idx = hop2_doc_local_idx.reshape(len(queries), 50, hop2_beam_size)

retrieved_results = {}
for idx, qid in enumerate(tqdm(queries.keys(), desc="Collecting results")):
    # 5.3.1 sort scores of the second-hop retrieval docs
    search_scores = path_scores[idx]
    ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],\
        (hop1_beam_size, hop2_beam_size))).transpose()

    hop1_docs = list(hop1_result[qid].keys())
    # hop2_docs = list(hop2_result[qid].keys())

    # 5.3.2 record retrieval results
    example_retrieved_corpus_ids = list(hop1_result[qid].keys())[:50]
    example_hop1_corpus_doc_ids = []
    example_path_corpus_doc_ids = []

    valid_counter = 0
    i = -1
    while valid_counter < 50:
        i += 1

        # get path info
        path_ids = ranked_pairs[i]
        hop1_corpus_doc_id = hop1_docs[path_ids[0]]
        # hop2_corpus_doc_id = hop1_docs[path_ids[0]]
        hop2_corpus_doc_id = faiss_search.rev_mapping[int(hop2_doc_local_idx[idx, path_ids[0], path_ids[1]])]

        # record path info
        _path = set([hop1_corpus_doc_id, hop2_corpus_doc_id])

        # skip if overlap
        if _path in example_path_corpus_doc_ids:
            continue

        example_retrieved_corpus_ids += [hop1_corpus_doc_id, hop2_corpus_doc_id]
        example_hop1_corpus_doc_ids.append(hop1_corpus_doc_id)
        example_path_corpus_doc_ids.append(_path)
        valid_counter += 1

    retrieved_results[qid] = MdrRecoder(
        example_retrieved_corpus_ids,
        example_hop1_corpus_doc_ids,
        example_path_corpus_doc_ids,
    )

qids_list = list(queries.keys())

eval_res = {}
for qid in queries.keys():
    p_recall, p_em = 0, 0
    labels = list(test_raw[qid].get('sps').keys())
    mdr_res = retrieved_results[qid]
    sp_covered = [l_i in mdr_res.retrieved_corpus_ids for l_i in labels]
    if np.sum(sp_covered) > 0:
        p_recall = 1
    if np.sum(sp_covered) == len(sp_covered):
        p_em = 1
    eval_res[qid] = {
        "recall@100":p_recall,
        "path_cover":p_em
    }
    

import numpy as np
r, em = [], []
for vs in eval_res.values():
    r.append(vs['recall@100'])
    em.append(vs['path_cover'])
print("recall:", np.mean(r))
print("em:", np.mean(em))

utils.save_pickle(Path(hop1_res_path)/ f"mdr_retrieval_res.pkl", retrieved_results)


def mdr_eval(res_name, hop1_beam, hop2_beam, pairs_num=50):
    hop1_res_path = args.hop1_res_path
    print("hop1 path: ", hop1_res_path)
    faiss_path = args.faiss_path
    print("load faiss: ", faiss_path)
    faiss_search = None
    faiss_search = beir_utils.MdrFaissSearch(dmodel, batch_size=512, use_gpu=False)
    faiss_search.index = None
    faiss_search.load(input_dir=faiss_path)

    print("load hop1 res")
    hop1_result, hop1_socres = faiss_search._load_results_and_scores(hop1_res_path, score_name="scores.pkl")
    print("load hop2 res")
    hop2_result = utils.load_pickle(Path(hop1_res_path)/ f"hop2_retrieval_res_large.pkl")
    hop2_socres = utils.load_pickle(Path(hop1_res_path)/ f"hop2_retrieval_scores_large.pkl")
    hop2_doc_local_idx = utils.load_pickle(Path(hop1_res_path)/ f"hop2_retrieval_local_ids_large.pkl")

    hop1_socres_ = hop1_socres[:, :hop1_beam]
    hop2_socres_ = hop2_socres.reshape(len(queries), 50, 100)
    hop2_socres_ = hop2_socres_[:,:hop1_beam,:hop2_beam]
    #hop2_doc_local_idx = hop2_doc_local_idx.reshape(len(queries), 50, 100)
    hop2_doc_local_idx = hop2_doc_local_idx.reshape(len(queries), 50, 100)

    path_scores = np.expand_dims(hop1_socres_, axis=2) * hop2_socres_

    print('Path scores shape',path_scores.shape)

    hop1_doc_local_idx = []
    for qid in list(queries.keys()):
        _res = []
        for j in hop1_result[qid].keys():
            _res.append(j)
        hop1_doc_local_idx.append(_res)
    hop1_doc_local_idx = np.array(hop1_doc_local_idx)

    retrieved_results = {}
    for idx, qid in enumerate(tqdm(queries.keys(), desc="Collecting results")):
        # 5.3.1 sort scores of the second-hop retrieval docs
        search_scores = path_scores[idx]
        ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],\
            (50, 20))).transpose()

        hop1_docs = list(hop1_result[qid].keys())
        # hop2_docs = list(hop2_result[qid].keys())

        # 5.3.2 record retrieval results
        example_retrieved_corpus_ids = list(hop1_result[qid].keys())[:hop1_beam]
        example_hop1_corpus_doc_ids = []
        example_path_corpus_doc_ids = []

        valid_counter = 0
        i = -1
        while valid_counter < pairs_num and i < ranked_pairs.shape[0]:
            i += 1

            # get path info
            path_ids = ranked_pairs[i]
            hop1_corpus_doc_id = hop1_docs[path_ids[0]]
            # hop2_corpus_doc_id = hop1_docs[path_ids[0]]
            hop2_corpus_doc_id = faiss_search.rev_mapping[int(hop2_doc_local_idx[idx, path_ids[0], path_ids[1]])]

            # record path info
            _path = set([hop1_corpus_doc_id, hop2_corpus_doc_id])

            # skip if overlap
            if _path in example_path_corpus_doc_ids:
                continue

            example_retrieved_corpus_ids += [hop1_corpus_doc_id, hop2_corpus_doc_id]
            example_hop1_corpus_doc_ids.append(hop1_corpus_doc_id)
            example_path_corpus_doc_ids.append(_path)
            valid_counter += 1

        MdrRecoder = namedtuple("MdrRecoder", ["retrieved_corpus_ids", "hop1_corpus_doc_ids", "path_corpus_doc_ids"])

        retrieved_results[qid] = MdrRecoder(
            example_retrieved_corpus_ids,
            example_hop1_corpus_doc_ids,
            example_path_corpus_doc_ids,
        )

    eval_res = {}
    r, em = [], []
    RECALL = f"recall@{2*pairs_num}"
    for qid in queries.keys():
        p_recall, p_em = 0, 0
        labels = list(test_raw[qid].get('sps').keys())
        mdr_res = retrieved_results[qid]
        sp_covered = [l_i in mdr_res.retrieved_corpus_ids for l_i in labels]
        if np.sum(sp_covered) > 0:
            p_recall = 1
        if np.sum(sp_covered) == len(sp_covered):
            p_em = 1
        eval_res[qid] = {
            RECALL:p_recall,
            "path_cover":p_em
        }
        r.append(p_recall)
        em.append(p_em)
    recall_res = np.mean(r)
    em_res = np.mean(em)
    print(f"{RECALL}: {recall_res}, path cover: {em_res}")
    utils.save_pickle(Path(hop1_res_path)/ f"mdrs_beams_{hop1_beam}_{hop2_beam}_top{pairs_num}_pairs.pkl", eval_res)

paris_list = [1,10,25,50]
for ppn in paris_list:
    print("=======================")
    mdr_eval("mopo_s", 10, 20, ppn)
