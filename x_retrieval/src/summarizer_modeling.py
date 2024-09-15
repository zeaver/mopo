import logging
import json
import csv
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import argparse

from collections import OrderedDict
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model, PeftModel

import sys
sys.path.append(r"x_retrieval")

from TEMPLATE import SUMMARY_INSTRUCTION, SUMMARY_TEMPLATE


parser = argparse.ArgumentParser()
parser.add_argument("--Result_path", default='', type=str, help=" Hop1 Result path")
parser.add_argument("--Save_path", default='', type=str, help="Input Save path")

args = parser.parse_args()

hop1_res_path = args.Result_path
hop2_input_path = args.Save_path

@dataclass
class BaseSumConfig:
    model_local_path: str = None
    max_length: int = None
    batchsize: int = None
    device: int = None


class BaseSummarizer(ABC):
    def __init__(self, config: BaseSumConfig):
        self.config = config
        self.device = config.device
        self._load_model_tokenizer()

    @staticmethod
    def cat_para(para):
        _t = "- " + para["title"].strip()
        _c = "- " + para["text"].strip()
        return f"{_t}\n{_c}"

    @abstractmethod
    def _load_model_tokenizer(self, **kwargs):
        pass

    @abstractmethod
    def generate_next_iter_info(
        self, 
        pre_queries: Dict[str, str],
        pre_retrieval_infos: Dict[str, OrderedDict[str, float]],
        corpus: Dict[str, Dict[str, str]],
        raw_test_info: Dict[str, Any],
        **kwargs
        ):
        """
        summarize pre information: query_{t-1} + doc_{t-1} --> query_{t}
        """
        pass

@dataclass
class QwenSumConfig(BaseSumConfig):
    adapter_path: str = None
    instruction: str = None
    template: str = None
    mapping_path: str = None

class QwenSummarizer(BaseSummarizer):
    def __init__(self, 
                 qwen_sum_config:QwenSumConfig):
        super().__init__(qwen_sum_config)
        self._prepare_model_prefix()
        # self._load_mapping()

    def _prepare_model_prefix(self):
        '''
            only for chat template
        '''
        self.instruction = self.config.instruction
        self.template = self.config.template

        # Mistral model
        # self.instruction = "[INST] " + self.instruction
        # self.template += " [/INST]"

        # Qwen
        _sys_info = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        self.instruction = _sys_info + "<|im_start|>user\n" + self.instruction
        self.template += "<|im_end|>\n<|im_start|>assistant\n"

    def _load_model_tokenizer(self):
        # model config and tokenizer
        # different model tokenizer has different init processing
        # espeically the eos or pad token init
        self.base_mode_config = AutoConfig.from_pretrained(self.config.model_local_path)
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_local_path, use_fast=True, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Summarizer's tokenizer padding side: left")

        # load base model and hooked peft module
        # Not recommended PeftModel(base_model, peft_id)
        model = AutoModelForCausalLM.from_pretrained(self.config.model_local_path, torch_dtype=torch.float16, device_map="auto", low_cpu_mem_usage=True)
        lora_config = LoraConfig.from_pretrained(self.config.adapter_path)
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def _prepare_prefix_kvcache(self):
        _prefix_input = self.tokenizer(self.instruction, return_tensors="pt").to(self.device)
        _prefix_output = self.model(**_prefix_input)
        return _prefix_output.past_key_values, _prefix_input.input_ids

    # def _load_mapping(self):
    #     self.mapping = load_tsv_to_dict(self.config.mapping_path, header=True)
    #     self.rev_mapping = {v: k for k, v in self.mapping.items()}

    def _cutoff_doc(self, text):
        _token_ids = self.tokenizer(text).input_ids
        if len(_token_ids) > 300:
            _token_ids = _token_ids[:300]

        cutoff_text = self.tokenizer.decode(_token_ids)

    def _prepare_next_query_input(
        self, 
        pre_queries: Dict[str, str],
        pre_retrieval_infos: Dict[str, OrderedDict[str, float]],
        corpus: Dict[str, Dict[str, str]],
        ):
        __pre_qids = set(pre_queries.keys())
        __input_qids = set(pre_retrieval_infos.keys())
        assert __pre_qids == __input_qids

        # step1: reshape queries and prepare prompts for summarizer
        _next_queries=[]
        for qid in tqdm(pre_queries, desc="Preparing inputs"):
            retrieved_doc_corpus_ids = pre_retrieval_infos[qid].keys()
            q_string = pre_queries[qid]
            for i, _corpus_doc_id in enumerate(retrieved_doc_corpus_ids):
                if i < 50:
                    _new_qid = f"{qid}_beam{i}"
                    retrieved_doc_string = self._cutoff_doc(self.cat_para(corpus[_corpus_doc_id]))
                    inst = self.instruction + self.template.format(context=retrieved_doc_string, question=q_string)
                    _tmp = {
                        "_id": _new_qid,
                        "hop1_doc_id": _corpus_doc_id,
                        "hop2_inst": inst
                    }
                    _next_queries.append(_tmp)
                else:
                    break

        return _next_queries

    def generate_next_iter_info(
        self, 
        pre_queries: Dict[str, str],
        pre_retrieval_infos: Dict[str, OrderedDict[str, float]],
        corpus: Dict[str, Dict[str, str]],
        ):
        """
        summarize pre information: query_{t-1} + doc_{t-1} --> query_{t}
        """
        next_input_dict = self._prepare_next_query_input(pre_queries, pre_retrieval_infos, corpus)
        pass

    def batch_generate(
        self,
        datasets: List[Dict[str, str]],
        save_path: str,
        save_freq: int=10
        ):
        bs = self.config.batchsize
        data_size = len(datasets)
        with open(save_path, "a") as f:
            _res = []
            for i in range(0, data_size, bs):
                # try:
                batch_infos = datasets[i : min(data_size, i + bs)]
                batch_input_texts = [x["hop2_inst"] for x in batch_infos]
                batch_inputs = self.tokenizer(
                    batch_input_texts,
                    truncation=True,
                    padding=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                ).to(self.device)
                prompt_len = batch_inputs.input_ids.size(-1)
                generate_ids = self.model.generate(
                    **batch_inputs,
                    max_length=864,
                    repetition_penalty=1.1,
                    top_p=0.8,
                    do_sample=True,
                )
                output_text = self.tokenizer.batch_decode(
                    generate_ids[:,prompt_len:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                for x, y in zip(batch_infos, output_text):
                    x.update({"hop2_query": y})
                    x.pop('hop2_inst')
                _res += batch_infos
                if i % save_freq == 0:
                    f.write(json.dumps(_res))
                    _res = []
                # except:
                #     pass

@dataclass
class T5SumConfig(BaseSumConfig):
    from TEMPLATE import FLAN_T5_INSTRUCTION
    template: str = FLAN_T5_INSTRUCTION

class T5Summarizer(BaseSummarizer):
    def __init__(self, 
                 t5_sum_config: T5SumConfig):
        super().__init__(t5_sum_config)
        self.template = self.config.template

    def _load_model_tokenizer(self):
        self.base_mode_config = AutoConfig.from_pretrained(self.config.model_local_path)
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_local_path, use_fast=True
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_local_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = tokenizer
        self.model = model.to(self.device)

    def _prepare_next_query_input(
        self, 
        pre_queries: Dict[str, str],
        pre_retrieval_infos: Dict[str, OrderedDict[str, float]],
        corpus: Dict[str, Dict[str, str]],
        queries_save_path: str,
        beam_num: int = 50
        ):

        assert queries_save_path.endswith("jsonl")
        if Path(queries_save_path).exists():
            return read_jsonl(queries_save_path)

        __pre_qids = set(pre_queries.keys())
        __input_qids = set(pre_retrieval_infos.keys())
        assert __pre_qids == __input_qids

        # reshape queries and prepare prompts for summarizer
        _next_queries=[]
        for qid in tqdm(pre_queries, desc="Preparing inputs"):
            retrieved_doc_corpus_ids = pre_retrieval_infos[qid].keys()
            q_string = pre_queries[qid]
            for i, _corpus_doc_id in enumerate(retrieved_doc_corpus_ids):
                if i < beam_num:
                    _new_qid = f"{qid}_beam{i}"
                    retrieved_doc_string = self.cat_para(corpus[_corpus_doc_id])
                    inst = self.template.format(context=retrieved_doc_string, question=q_string)
                    _tmp = {
                        "_id": _new_qid,
                        "hop1_doc_id": _corpus_doc_id,
                        "hop2_inst": inst
                    }
                    _next_queries.append(_tmp)
                else:
                    break
        
        save_jsonl(queries_save_path, _next_queries)

        return _next_queries
    
    def generate_next_iter_info(self, **kwags):
        pass


def read_json(json_data_path: Path):
    with Path(str(json_data_path)).open("r", encoding="utf8") as file:
        data = json.load(file)
    return data

def read_jsonl(json_data_path: Path):
    with Path(str(json_data_path)).open("r", encoding="utf8") as file:
        return [json.loads(line.strip()) for line in file.readlines()]

def save_jsonl(my_path, my_data):
    with Path(str(my_path)).open("w", encoding="utf-8") as fout:
        for d in my_data:
            json_str = json.dumps(d)
            fout.write(json_str + '\n')

def load_tsv_to_dict(input_path, header=True):
    
    mappings = {}
    reader = csv.reader(open(input_path, encoding="utf-8"), 
                            delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    if header: next(reader)
    for row in reader: 
        mappings[row[0]] = int(row[1])
    
    return mappings

def test_prefix_kvcache():
    test_config = QwenSumConfig(
        model_local_path="download/qwen_1.5_0.5b",
        max_length=1024,
        batchsize=4,
        device="cuda",
        adapter_path=r"sum_output/qwen_1.5_0.5b_lora_0508",
        instruction=SUMMARY_INSTRUCTION,
        template=SUMMARY_TEMPLATE
    )
    summary_runner = QwenSummarizer(test_config)
    test_data = read_jsonl(r"x_retrieval/datasets/mdr_hotpot/train.jsonl")[20000:20005]
    prefix_kv_cache, prefix_ids = summary_runner._prepare_prefix_kvcache()
    res = []
    for d in tqdm(test_data):
        retrieved_doc_string = summary_runner.cat_para(d["second_hop_doc"])
        input_text = summary_runner.template.format(context=retrieved_doc_string, question=d.get("first_hop_query"))
        input_text_ids = summary_runner.tokenizer(input_text, return_tensors="pt").input_ids.to("cuda")
        all_input_ids = torch.cat((prefix_ids, input_text_ids), dim=-1)
        all_attention_mask = torch.ones_like(all_input_ids)
        generate_ids = summary_runner.model.generate(all_input_ids, attention_mask = all_attention_mask, past_key_values=prefix_kv_cache, max_length=2048)
        # output_text = summary_runner.tokenizer.decode(generate_ids[0][all_input_ids.size(-1):], skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output_text = summary_runner.tokenizer.decode(generate_ids[0])
        _tmp_res = {
            "_id":d["raw_id"],
            "generated_first_hop_summary": output_text
        }
        res.append(_tmp_res)
    save_jsonl(r"x_retrieval/experiments/sum_predict/test_predict.jsonl", res)



def test_t5_sum():
    # raw data
    from beir.datasets.data_loader import GenericDataLoader
    beir_data_path = r"x_retrieval/datasets/beir_hotpot"
    corpus, queries, _ = GenericDataLoader(data_folder=beir_data_path).load()

    # load hop1 res
    import pickle
    
    with open(hop1_res_path, "rb") as f:
        hop1_res = pickle.load(f)

    # load summarization model
    test_t5_config = T5SumConfig(
        model_local_path="download/flan_t5_large",
        max_length=60,
        batchsize=4,
        device="cuda"
    )
    t5_runner = T5Summarizer(test_t5_config)
    
    infra_input_path = hop2_input_path + "/t5_hop2_inputs.jsonl"
    t5_inputs = t5_runner._prepare_next_query_input(queries, hop1_res, corpus, infra_input_path)
    print("done")

if __name__ == "__main__":
    # test()

    test_t5_sum()
