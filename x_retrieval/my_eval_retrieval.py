import argparse
import copy
import json
import os
from pathlib import Path
import logging
from typing import Dict

import faiss
from faiss import write_index, read_index

import numpy as np
import torch
import time
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm



try:
    from src.contriever import Contriever, load_retriever
    from src.utils import AverageMeter, init_logger
except:
    pass

logger = logging.getLogger(__name__)


class EvalDataset(Dataset):
    def __init__(self, 
                 data_path,
                 candidates_path,
                 add_prefix = True,
                 doc_prefix="search_document",
                 query_prefix="search_query",
                 ):
        
        self.doc_prefix = doc_prefix
        self.query_prefix = query_prefix
        self.if_add_prefix = add_prefix

        self.load_data(data_path)
        self.candidates = self._load_candidates(candidates_path)

    def __len__(self):
        return len(self.id_list)

    def __getitem__(self, index):
        id_ = self.id_list[index]
        query = self.query_list[index] if not self.if_add_prefix else self._add_text_prefix(self.query_prefix, self.query_list[index])
        positive = []
        for posi in self.positive_list[index]:
            text = posi if not self.if_add_prefix else self._add_text_prefix(self.doc_prefix, posi)
            positive.append(text)
        # positive = self.positive_list[index] if not self.if_add_prefix else self._add_text_prefix(self.doc_prefix, self.positive_list[index])
        positive_ids = self.positive_id_list[index]
        return id_, query, positive, positive_ids

    def load_data(self, data_path):
        if str(data_path).endswith("json"):
            data_list = self._load_data_json(data_path)
        elif str(data_path).endswith("jsonl"):
            data_list = self._load_data_jsonl(data_path)
        else:
            raise ValueError(f"data path error, input: {data_path} is illegal")
        
        self.raw_data_list = data_list

        id_list = []
        query_list = []
        positive_list = []
        posi_ids = []

        for d in data_list:
            id_list.append(d.get("local_id"))
            query_list.append(d.get("question"))
            positive_list.append(d.get("positives"))
            posi_ids.append(d.get("positive_ids"))
        
        self.id_list = id_list
        self.query_list = query_list
        self.positive_list = positive_list
        self.positive_id_list = posi_ids

    def _load_data_json(self, path):
        with open(path, "r") as fin:
            examples = json.load(fin)
        return examples

    def _load_data_jsonl(self, path):
        examples = []
        with open(path, "r") as fin:
            for line in fin:
                example = json.loads(line)
                examples.append(example)
        return examples

    def _add_text_prefix(self, text_prefix, text):
        return f"{text_prefix}: {text}"
    
    def _load_candidates(self, data_path):
        if str(data_path).endswith("json"):
            data_list = self._load_data_json(data_path)
        elif str(data_path).endswith("jsonl"):
            data_list = self._load_data_jsonl(data_path)
        else:
            raise ValueError(f"data path error, input: {data_path} is illegal")
        candidates = []
        for d in data_list:
            if not self.if_add_prefix:
                _passage = d.get("text")
            else:
                _passage = self._add_text_prefix(self.doc_prefix, d.get("text"))
            candidates.append(_passage)
        return candidates

    @staticmethod
    def collate(batch):
        ids = [item[0] for item in batch]
        querys = [item[1] for item in batch]
        positives = [item[2] for item in batch]
        positives_ids = [item[3] for item in batch]
        return ids, querys, positives, positives_ids
    
    def demo_logger(self, in_logger):
        demo_query = "\n".join([self[i][1] for i in range(2)])
        demo_candidates = "\n".join(self.candidates[:2])
        try:
            in_logger.info(f"\nQuery demo:\n{demo_query}")
            in_logger.info(f"\nCorpus demo:\n{demo_candidates}")
        except:
            print(f"logger failed in {self.__class__.__name__}")


class EvalRunner:
    def __init__(self,
                 model_ckpt_path,
                 eval_dataset,
                 eval_args,
                 device):
        self.retriever, self.tokenizer, self.model_id = load_retriever(eval_args.model_name, model_ckpt_path)
        self.eval_dataset = eval_dataset
        self.args = eval_args
        self.load_dataset_loader()
        self.device = device
        self.retriever.to(device)

    def load_dataset_loader(self):
        self.eval_dataset_loader = DataLoader(dataset=self.eval_dataset,
                                              batch_size=self.args.eval_batch_size, 
                                              collate_fn=self.eval_dataset.collate)

    def build_corpus_index(self):
        model_eval_dir = Path(self.args.model_id_eval_path)
        index_path = model_eval_dir / "corpus.index"
        if index_path.exists():
            self.index = read_index(str(index_path))
        else:
            all_ctx_vector = []
            for mini_batch in tqdm(range(0, len(self.eval_dataset.candidates), self.args.eval_batch_size)):
                contexts = self.eval_dataset.candidates[mini_batch:mini_batch + self.args.eval_batch_size]
                tokenizer_outputs = self.tokenizer.batch_encode_plus(
                    contexts,
                    padding=True, 
                    return_tensors='pt', 
                    max_length=self.args.context_length, 
                    truncation=True
                )
                context_input_ids = tokenizer_outputs.input_ids.to(self.device)
                context_attention_mask = tokenizer_outputs.attention_mask.to(self.device)
                sub_ctx_vector = self.retriever(context_input_ids, context_attention_mask, normalize=self.args.norm_doc).cpu().detach().numpy()
                all_ctx_vector.append(sub_ctx_vector)

            all_ctx_vector = np.concatenate(all_ctx_vector, axis=0)
            all_ctx_vector = np.array(all_ctx_vector).astype('float32')
            index = faiss.IndexFlatIP(all_ctx_vector.shape[-1])
            index.add(all_ctx_vector)
            self.index = index
            write_index(index, str(index_path))

    @staticmethod
    def init_results_dict():
        results = {
            'id_list': [],
            'outputs': [],
            'targets': []
        }
        return copy.deepcopy(results)

    @staticmethod
    def init_meters_dict():
        meters = {
            'R@1': AverageMeter('R@1', ':6.4f'),
            'R@5': AverageMeter('R@5', ':6.4f'),
            'R@20': AverageMeter('R@20', ':6.4f'),
            'R@100': AverageMeter('R@100', ':6.4f'),
            'MRR@5': AverageMeter('MRR@5', ':6.4f'),
        }
        return copy.deepcopy(meters)

    @staticmethod
    def measure_result(result_dict, meters):
        outputs = result_dict['outputs']
        targets = result_dict['targets']
        for output, target in zip(outputs, targets):
            # r1 = 1 if target == output[0] else 0
            r1 = 1 if output[0] in target else 0
            meters['R@1'].update(r1)
            target = set(target)
            r5 = 1 if len(target & set(output[:5])) > 0 else 0
            # r5 = 1 if target in output[:5] else 0
            meters['R@5'].update(r5)
            # r20 = 1 if target in output[:20] else 0
            r20 = 1 if len(target & set(output[:20])) > 0 else 0
            meters['R@20'].update(r20)
            # r100 = 1 if target in output else 0
            r100 = 1 if len(target & set(output[:100])) > 0 else 0
            meters['R@100'].update(r100)
            if r5 == 1:
                l5 = np.array([output.index(int(x)) for x in list(target & set(output[:5]))]).min()
                meters['MRR@5'].update(1 / (l5 + 1))
            else:
                meters['MRR@5'].update(0)


    def inference_step(self, payload, result_dict):
        ids, querys, positives, positives_ids = payload

        tokenizer_outputs = self.tokenizer.batch_encode_plus(querys, 
                                                             padding=True, 
                                                             return_tensors='pt',
                                                             max_length=self.args.query_length, 
                                                             truncation=True)
        input_ids = tokenizer_outputs.input_ids.to(self.device)
        attention_mask = tokenizer_outputs.attention_mask.to(self.device)
        query_vector = self.retriever(input_ids, attention_mask, normalize=self.args.norm_query).cpu().numpy().astype('float32')

        D, I = self.index.search(query_vector, 100)
        result_dict['id_list'] += ids
        result_dict['targets'] += positives_ids
        result_dict['outputs'] += I.tolist()

    @torch.no_grad()
    def inference(self):
        batch_time = AverageMeter('Time', ':6.3f')
        meters = self.init_meters_dict()
        result_dict = self.init_results_dict()

        self.retriever.eval()
        end = time.time()
        self.build_corpus_index()
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        except:
            pass

        for idx, payload in enumerate(tqdm(self.eval_dataset_loader)):
            self.inference_step(payload, result_dict)
            batch_time.update(time.time() - end)
            end = time.time()


        with open(f'{self.args.output_dir}/result.json', 'w') as f:
            json.dump(result_dict, f)
        self.measure_result(result_dict, meters)
        with open(f'{self.args.output_dir}/performance.json', 'w') as f:
            eval_res = {field_name: meter.avg for field_name, meter in meters.items()}
            json.dump(eval_res, f)
        self.eval_first_hop(result_dict)

    def eval_first_hop(self, res_dict_or_path):

        # load result
        if isinstance(res_dict_or_path, str):
            assert str(res_dict_or_path).endswith("json")
            with open(str(res_dict_or_path), "r", encoding='utf8') as fin:
                res_dict = json.load(fin)
            _save_path = Path(res_dict_or_path).parent / "first_hop_performance.json"
        elif isinstance(res_dict_or_path, Dict):
            res_dict = res_dict_or_path
            _save_path = f'{self.args.output_dir}/first_hop_performance.json'
        else:
            raise ValueError(f"res_dict_or_path is illegal, get {res_dict_or_path} but expect path or Dict")
        
        idx_list = res_dict.get('id_list')
        outputs = res_dict.get('outputs')
        new_targets = []
        
        for idx, output in zip(idx_list, outputs):
            idx = int(idx)
            target = self.eval_dataset[idx][-1]
            if self.eval_dataset.raw_data_list[idx]["type"] != "comparison":
                new_targets.append([target[0]])
            else:
                new_targets.append(target)
        
        new_res = {
            "id_list": idx_list,
            "outputs": outputs,
            "targets": new_targets
        }

        new_meters = self.init_meters_dict()
        self.measure_result(new_res, new_meters)


        with open(_save_path, 'w') as f:
            eval_res = {field_name: meter.avg for field_name, meter in new_meters.items()}
            json.dump(eval_res, f)

def main(parser):
    opt = parser.parse_args()
    opt.model_id_eval_path = str( Path(opt.output_dir) / Path(opt.model_path).name )
    opt.output_dir = os.path.join(opt.model_id_eval_path, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))

    if not Path(opt.output_dir).exists():
        Path(opt.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"[info] mkdir: {opt.output_dir}")
    
    # init logger
    message = ""
    for k, v in sorted(vars(opt).items()):
        comment = ""
        default = parser.get_default(k)
        if v != default:
            comment = f"\t[default: %s]" % str(default)
        message += f"{str(k):>40}: {str(v):<40}{comment}\n"
    print(message, flush=True)
    file_name = os.path.join(opt.output_dir, "opt.txt")
    with open(file_name, "wt") as opt_file:
        opt_file.write(message)
        opt_file.write("\n")    
    
    init_logger(opt)
    
    
    opt.devices = "cuda"

    eval_dataset = EvalDataset(data_path = opt.eval_dataset_path,
                               candidates_path = opt.corpus_dataset_path,
                               add_prefix = opt.if_add_prefix,
                               doc_prefix = opt.doc_prefix,
                               query_prefix = opt.query_prefix
                               )
    
    eval_dataset.demo_logger(logger)
    eval_runner = EvalRunner(model_ckpt_path=opt.model_path,
                             eval_dataset=eval_dataset,
                             eval_args=opt,
                             device="cuda"
                             )
    eval_runner.inference()


def test_first_hop(parser):
    opt = parser.parse_args()
    eval_dataset = EvalDataset(data_path = opt.eval_dataset_path,
                               candidates_path = opt.corpus_dataset_path,
                               add_prefix = opt.if_add_prefix
                               )
    eval_runner = EvalRunner(model_ckpt_path=opt.model_path,
                             eval_dataset=eval_dataset,
                             eval_args=opt,
                             device="cuda"
                             )
    eval_runner.eval_first_hop(r"/x_retrieval/experiments/eval_output/contriever_msmarco/2024_03_27_08_17_55/result.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--corpus_dataset_path", type=str, default="/x_retrieval/datasets/mdr_hotpot/mdr_dev_corpus_neg20.jsonl")
    parser.add_argument("--eval_dataset_path", type=str, default="/x_retrieval/datasets/mdr_hotpot/mdr_dev_neg20.jsonl")


    parser.add_argument("--model_name", type=str, default=r"e5" )
    parser.add_argument("--model_path", type=str, default=r"/huggingface_download/e5_base_v2" )
    parser.add_argument("--output_dir", type=str, default="/x_retrieval/experiments/eval_local_output")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Batch size for the passage encoder forward pass")

    parser.add_argument("--if_add_prefix", type=bool, default=True)
    parser.add_argument("--doc_prefix", type=str, default="passage")
    parser.add_argument("--query_prefix", type=str, default="query")
    # parser.add_argument("--statement_prefix", type=str, default="search_statement")
    parser.add_argument("--norm_query", type=int, default=1, choices=[0,1])
    parser.add_argument("--norm_doc", type=int, default=1, choices=[0,1])
    parser.add_argument("--query_length", type=int, default=100)
    parser.add_argument("--context_length", type=int, default=350)

    main(parser)
    # test_first_hop(parser)

    

