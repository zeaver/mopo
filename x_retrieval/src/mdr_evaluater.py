import pdb
import os
import time
import sys
from typing import Dict, List
import logging
import json
from tqdm import tqdm
import numpy as np
from pathlib import Path
from collections import defaultdict, namedtuple

from src.options import Options
from src import beir_utils, dist_utils, contriever, utils

# import train

os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)

# recoder
# (1) total retrieved docs
# (2) path
# (3) firt-hop retrieval performance
MdrRecoder = namedtuple("MdrRecoder", ["retrieved_corpus_ids", "hop1_corpus_doc_ids", "path_corpus_doc_ids"])

def main():
    logger.info("Start")

    options = Options()
    opt = options.parse()
    # opt.model_path = r"/home/student2021/a/x_retrieval/experiments/eval_models/mdr_proposal"
    datasetname = opt.eval_datasets[0]

    _t = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    _model_name = Path(opt.model_path).name

    opt.eval_output_dir = os.path.join(r"x_retrieval/experiments/beir_eval_output", datasetname, _model_name, _t)
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

    logger.info(f"Start evaluation")

    # step1: Init
    # 1.1 load dataset
    data_path = os.path.join(opt.eval_datasets_dir, datasetname)
    corpus, queries, qrels, test_raw = beir_utils.MdrBeirDataloader(data_folder=data_path).load(split="test")

    # 1.2 init dense model
    print(type(opt.model_path))
    eval_mode, tokenizer, retriever_model_id = contriever.load_retriever(
        opt.model_name, opt.model_path, raw_model_id=opt.raw_model_id
    )
    opt.retriever_model_id = retriever_model_id
    eval_mode.cuda()
    dmodel = beir_utils.DenseEncoderModel(
        query_encoder=eval_mode,
        doc_encoder=eval_mode,
        tokenizer=tokenizer,
        add_prefix=opt.if_add_prefix,
        query_prefix=opt.query_prefix,
        doc_prefix=opt.doc_prefix,
        norm_doc=opt.norm_doc,
        norm_query=opt.norm_query,
        )

    # 1.3 init search engine
    faiss_search = beir_utils.MdrFaissSearch(dmodel, batch_size=512, use_gpu=opt.faiss_gpu)
    # 1.3.1 save corpus embedding and _id mapping dicts
    # faiss_search.index(corpus)
    # faiss_search.save(str(Path(opt.eval_output_dir).parent))
    # 1.3.2 laod corpus embedding and _id mapping dicts
    faiss_search.load(input_dir=str(Path(opt.eval_output_dir).parent))

    # step2: Running first hop retrieval
    # or directly load them
    # hop1_result, hop1_socres, hop1_doc_local_idx = faiss_search.hop_search(corpus, queries, opt.hop1_beam)
    hop1_result, hop1_socres = faiss_search._load_results_and_scores(opt.hop1_res_path)

    # step3: Running query generation for second iter retrieval
    # or directly load it
    # next_queries = query_summarizer.generate_next_iter_info(
    #     queries,
    #     corpus,
    #     hop1_result,
    #     opt.hop2_beam
    # )

    hop2_quries_dicts = faiss_search._load_hop2_queries(opt.hop1_res_path)
    hop2_queries, hop2_info = faiss_search._resort_hop2(hop2_quries_dicts, queries, 50)

    # step4: Running second hop retrieval
    hop2_result, hop2_socres, hop2_doc_local_idx = faiss_search.hop_search(corpus, next_queries, opt.hop2_beam)

    # step5: Computing evaluation results
    # 5.1 reshape scores and ids(local idx in list)
    hop2_socres = hop2_socres.reshape(len(queries), opt.hop1_beam, opt.hop2_beam)
    hop2_doc_local_idx = hop2_doc_local_idx.reshape(len(queries), opt.hop1_beam, opt.hop2_beam)

    # 5.2 aggregate path scores
    path_scores = np.expand_dims(hop1_socres, axis=2) + hop2_socres

    # 5.3 traverse every query result
    # {qid: q_mdr_result}
    retrieved_results: Dict[str, MdrRecoder] = {}
    for idx, qid in enumerate(tqdm(queries.keys(), desc="Collecting results")):
        # 5.3.1 sort scores of the second-hop retrieval docs
        search_scores = path_scores[idx]
        ranked_pairs = np.vstack(np.unravel_index(np.argsort(search_scores.ravel())[::-1],\
            (opt.hop1_beam, opt.hop2_beam))).transpose()

        # 5.3.2 record retrieval results
        example_retrieved_corpus_ids = list(hop1_result[qid].keys())
        example_hop1_corpus_doc_ids = []
        example_path_corpus_doc_ids = []

        valid_counter = 0
        i = -1
        while valid_counter < opt.topk:
            i += 1

            # get path info
            path_ids = ranked_pairs[i]
            hop1_corpus_doc_id = faiss_search.rev_mapping[hop1_doc_local_idx[idx, path_ids[0]]]
            hop2_corpus_doc_id = faiss_search.rev_mapping[hop2_doc_local_idx[idx, path_ids[0], path_ids[1]]]

            # record path info
            example_path_corpus_doc_ids.append([hop1_corpus_doc_id, hop2_corpus_doc_id])

            # skip if overlap
            if hop1_corpus_doc_id in example_retrieved_corpus_ids and hop2_corpus_doc_id in example_retrieved_corpus_ids:
                continue

            
            example_retrieved_corpus_ids += [hop1_corpus_doc_id, hop2_corpus_doc_id]
            example_hop1_corpus_doc_ids.append(hop1_corpus_doc_id)
            

        retrieved_results[qid] = MdrRecoder(
            example_retrieved_corpus_ids,
            example_hop1_corpus_doc_ids,
            example_path_corpus_doc_ids,
        )


def mdr_evaluation(
    retrieval_results: Dict[str, MdrRecoder], 
    test_raw_info: Dict[str, Dict],
    topks: List[int]
) -> Dict[str, float]:
    """
    Evaluate two-hop iterative retrieval results
    - retrieval_results:
    - test_raw_info schema:
        ```
            - question
            - type: question type
            - _id: corpus id
            - answer
            - doc_ids: positive passage corpus ids
            - sps: supporting passages info
            - metadata
        ```
    """
    
    mdr_metrics = namedtuple(
        "mdr_metrics", ["qid", "type", "p_recall", "p_em", "recall_1", "path_covered"]
    )

    for qid in retrieval_results.keys():
        retrieval_res = retrieval_results[qid]
        labeled_info = test_raw_info[qid]
        supporting_corpus_ids = labeled_info["doc_ids"]
        q_type = labeled_info["type"]

    pass


if __name__ == "__main__":
    main()
