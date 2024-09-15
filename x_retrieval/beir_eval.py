import pdb
import os
import time
import sys
import torch
import logging
import json
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import FlatIPFaissSearch

from src.options import Options
from src import dist_utils, contriever
from src.beir_utils import DenseEncoderModel, MdrBeirDataloader, MdrFaissSearch

import pickle

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def main():
    logger.info("Start")

    options = Options()
    opt = options.parse()
    datasetname = opt.eval_datasets[0]

    _t = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    _model_name = Path(opt.model_path).name

    opt.eval_output_dir = os.path.join(r"x_retrieval/experiments/beir_eval_output", datasetname, _model_name, _t)
    print("eval_output_dir: ", opt.eval_output_dir)
    if not os.path.exists(opt.eval_output_dir):
        os.makedirs(opt.eval_output_dir, exist_ok=True)
        print("create dir: ", opt.eval_output_dir)

    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    file_handler = logging.FileHandler(filename=os.path.join(opt.eval_output_dir, "run.log"))
    handlers.append(file_handler)
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if dist_utils.is_main() else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )

    # WrapModel, _, _, _, _ = utils.load(contriever.Contriever, opt.model_path, opt)
    # retriever = WrapModel.get_encoder()
    print(type(opt.model_path))
    eval_mode, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_name, opt.model_path, raw_model_id=opt.raw_model_id)
    opt.retriever_model_id = retriever_model_id

    eval_mode.cuda()

    dmodel = DenseEncoderModel(
        query_encoder=eval_mode,
        doc_encoder=eval_mode,
        tokenizer=tokenizer,
        add_prefix=opt.if_add_prefix,
        query_prefix=opt.query_prefix,
        doc_prefix=opt.doc_prefix,
        norm_doc=opt.norm_doc,
        norm_query=opt.norm_query,
        )

    faiss_search = MdrFaissSearch(dmodel, batch_size=512, use_gpu=opt.faiss_gpu)
    # faiss_search = FlatIPFaissSearch(dmodel, batch_size=512, use_gpu=opt.faiss_gpu)

    splits = [
        # "sum_bridge_hop1",
        # "sum_bridge_hop2",
        # "sum_comparison_hop1",
        # "sum_comparison_hop2",
        # "sum_total_hop1",
        # "sum_total_hop2",
        # "base_bridge_hop1",
        # "base_bridge_hop2",
        # "base_comparison_hop1",
        # "base_comparison_hop2",
        # "base_total_hop1",
        # "base_total_hop2",
        # "gold_sum_bridge_hop2",
        # "gold_sum_comparison_hop2",
        # "gold_sum_total_hop2",
        "test"
    ]

    # load dataset
    data_path = os.path.join(opt.eval_datasets_dir, datasetname)
    res_path = Path(opt.eval_output_dir) / f"performance.json"
    data_loader = MdrBeirDataloader(data_folder=data_path, query_file="queries_added_sums.jsonl")
    #data_loader = GenericDataLoader(data_folder=data_path)
    corpus, queries, qrels = data_loader.load(split=splits[0])

    ## run 2 lines first
    faiss_search.index(corpus)
    faiss_search.save(str(Path(opt.eval_output_dir).parent))
    #return 

    metrics = {}
    faiss_search.load(input_dir=str(Path(opt.eval_output_dir).parent))
    retriever_eval = EvaluateRetrieval(faiss_search,  k_values = [1,2,3,5,10,20,50,100,1000], score_function="dot")

    for split_i in splits:
        queries, qrels = data_loader.reload_queries(split_i)
        results, res_score, _ = retriever_eval.retriever.hop_search(corpus, queries)
        ndcg, _map, recall, precision = retriever_eval.evaluate(qrels, results, retriever_eval.k_values)
        # for metric in (ndcg, _map, recall, precision, "mrr", "recall_cap", "hole"):
        #     if isinstance(metric, str):
        #         metric = retriever.evaluate_custom(qrels, results, retriever.k_values, metric=metric)
        #     for key, value in metric.items():
        #         metrics[key].append(value)

        recall = {k:v*100 for k,v in recall.items()}

        # my_recall = data_loader.eval_recall(retriever.k_values, results, qrels)
        _info = {
            "data_num": len(qrels.keys()),
            "beir_results": recall,
            # "my_recall": my_recall
        }
        metrics[split_i] = _info

        with open(Path(opt.eval_output_dir) / f"{_model_name}_retrieval_res.pkl", 'wb') as f:
            pickle.dump(results, f)
        with open(Path(opt.eval_output_dir) / f"{_model_name}_retrieval_score.pkl", 'wb') as f:
            pickle.dump(res_score, f)
        
        
    with open(res_path, "w") as fin:
        json.dump(metrics, fin, indent=4)

    # message = []
    # for metric in metrics.keys():
    #     message.append(f"{datasetname}/{metric}: {metrics[metric]:.2f}")
    # logger.info(" | ".join(message))
    # print(" | ".join(message))


if __name__ == "__main__":
    main()
