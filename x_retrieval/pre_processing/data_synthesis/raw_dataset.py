__all__ = ["RawHotpotConfig", "RawHotpot"]

import os
import random
from pathlib import Path
from typing import Dict, Generator
from dataclasses import dataclass, field

import json
from tqdm import tqdm
import numpy as np

try:
    from .normalize_text import normalize
except:
    import sys
    sys.path.append(str(Path(__file__).absolute().parent))
    from normalize_text import normalize

def read_json(json_data_path: Path):
    with Path(str(json_data_path)).open("r", encoding="utf8") as file:
        data = json.load(file)
    return data

def read_jsonl(json_data_path: Path):
    with Path(str(json_data_path)).open("r", encoding="utf8") as file:
        return [json.loads(line.strip()) for line in file.readlines()]

@dataclass
class RawHotpotConfig:
    mdr_hotpot_path: str = field(
        default='pre_processing/mdr_hotpot/mdr_train_mf_idx.jsonl',
        metadata={"help": "Path to MDR Hotpot dataset"}
    )
    mf_hotpot_path: str = field(
        default='pre_processing/multifactor_hotpot/train.json',
        metadata={"help": "Path to MF Hotpot dataset"}
    )
    output_dir: str = field(
        default='pre_processing/output',
        metadata={"help": "Directory to save the output"}
    )
    stage: str = field(
        default='summary',
        metadata={"help": ""}
    )
    summary_mode: int = field(
        default= 2,
        metadata={"help": ""}
    )
    if_extend_sp: bool = field(
        default= False,
        metadata={"help": ""}
    )

    def __post_init__(self):
        # convert relative to abs path string
        project_dir = Path(__file__).absolute().parent.parent.parent
        for field_name, field_value in self.__dict__.items():
            if field_name in ["mdr_hotpot_path", "mf_hotpot_path", "output_dir"]:
                abs_path = project_dir / field_value
                setattr(self, field_name, str(abs_path))
                assert Path(abs_path).is_file() or Path(abs_path).is_dir(), f"file: {abs_path} is not exist, please check it"
        
        assert self.stage in ["summary", "refine", "comparison", "full_answer"], \
            f"stage is supposed to be \"summarize\" or \"refine\", bue get {self.stage}"

class RawHotpot:
    def __init__(self, 
                 config: RawHotpotConfig, 
                 instruction: Dict,
                 logger):
        self.config = config
        self.logger = logger
        self.instruction = instruction[self.config.stage]

        # get data
        self.load_dataset()
        self.processed_mdr_list = self.load_processed_mdr_list()

    def __post_init__(self):
        pass
            
    def load_processed_mdr_list(self) -> np.ndarray:
        processed_list_path = Path(str(self.config.mdr_hotpot_path)).parent / "processed_list.npy"
        self.processed_list_path = processed_list_path
        self.logger.info(f"To found processed data index in {processed_list_path}")
        if not processed_list_path.is_file():
            self.logger.info("none processed example")
            return []
        else:
            processed_list = np.load(str(processed_list_path))
            self.logger.info("load processed example indexs of mdr dataset")
            return processed_list
        
    def get_example(self, 
                    start_idx: int=0,
                    end_idx: int=-1):
        if end_idx == -1 or end_idx > len(self.mdr_data):
            end_idx = len(self.mdr_data)
        self.logger.info(f"Start processing mdr hotpot dataset, range is **{start_idx}** to **{end_idx}**")
        for idx, sample in enumerate(tqdm(self.mdr_data[start_idx: end_idx])):
            if self.check_example_is_valid(idx, sample):
                sub_info:Dict = self.get_example_info(sample, self.mf_data[int(sample.get("multifactor_abs_id"))])
                yield int(idx) + start_idx, sample, sub_info

    def get_example_info(self, mdr_example: Dict, mf_example: Dict) -> Dict:
        if self.config.stage == "summary":
            res = self.get_example_info_for_summary(mdr_example, mf_example)
        elif self.config.stage == "refine":
            res = self.get_example_info_for_refine(mdr_example, mf_example)
        elif self.config.stage == "comparison":
            res = self.get_example_info_for_summary(mdr_example, mf_example)
            del res["question_type"]
            res["context"] = res["supporting_facts"]
            del res["supporting_facts"]
        elif self.config.stage == "full_answer":
            res = self.get_example_info_for_full_answer(mdr_example, mf_example)
        else:
            raise ValueError(f"Check config stage: {self.config.stage} is illegal")
        return res

    def get_example_info_for_summary(self, mdr_example: Dict, mf_example: Dict):
        # assert mdr_example.get("type").strip().lower() == mf_example.get("type").strip().lower()

        # get bridge paragraph title in mf exmaple
        # need know the data format of mdr hotpot
        if mdr_example["type"] == "comparison":
            random.shuffle(mdr_example["pos_paras"])
            start_para, bridge_para = mdr_example["pos_paras"]
        else:
            for para in mdr_example["pos_paras"]:
                if para["title"] != mdr_example["bridge"]:
                    start_para = para
                else:
                    bridge_para = para
        start_para_title = start_para.get("title")
        normed_start_para_title = normalize(start_para_title).lower()
        normed_match_dict = mdr_example["sp_title_mapping"]
        sp_title_in_mf = normed_match_dict[normed_start_para_title]

        # get supporting fact sentences
        # need know hotpotQA data format
        # 1. get all (title, doc) pairs, totally 20
        mf_paras_context_dict = {x[0]:x[1] for x in mf_example["context"]}

        # 2. get the supporting fact sentences index in the start paragraph 
        sp_sentences_idx = [x[1] for x in mf_example["supporting_facts"] if x[0]==sp_title_in_mf]

        # 3. extend the supporting fact sentences index
        sp_doc_sent_num = len(mf_paras_context_dict[sp_title_in_mf])
        if self.config.if_extend_sp:
            sp_sentences_idx = self._correct_sp_sentences_id(sp_sentences_idx, sp_doc_sent_num)

        # 4. get sp sentences
        sp_sentences = [mf_paras_context_dict[sp_title_in_mf][i].strip() for i in sp_sentences_idx]

        # 5. add bridge info
        if mdr_example["type"] == "bridge":
            _t = mf_example.get("type")
            _ent = mdr_example["bridge"]
            type_info = f"{_t}\n###BridgeInfo: {_ent}"
        else:
            type_info = mf_example.get("type")

        # 6. re-format for query
        context = [sp_title_in_mf] + sp_sentences
        context_in_prompt = "\n".join([f"- {x}" for x in context])
        if self.config.summary_mode == 1 or self.config.stage == "comparison":
            re_structure_info = {
                "question": mf_example.get("question"),
                "question_type": type_info,
                "supporting_facts": context_in_prompt,
            }
        else:
            re_structure_info = {
                "context": context_in_prompt,
                "statement": mf_example.get("full answer")
            }
        return re_structure_info

    def get_example_info_for_refine(self, summary_example: Dict, mf_example: Dict):
        summary_info = summary_example.get("gpt_info")
        return {
            "abstract": summary_info.get("abstract"),
            "context": summary_info.get("query_context"),
            "type_and_bridge_ent": summary_info.get("query_question")
        }
         
    def get_example_info_for_full_answer(self, mdr_example: Dict, mf_example: Dict):
        mf_paras_context_dict = {x[0]:x[1] for x in mf_example["context"]}
        def get_context(para):
            para_title = para.get("title")
            normed_para_title = normalize(para_title).lower()
            normed_match_dict = mdr_example["sp_title_mapping"]
            sp_title_in_mf = normed_match_dict[normed_para_title]
            sp_sentences_idx = [x[1] for x in mf_example["supporting_facts"] if x[0]==sp_title_in_mf]
            sp_sentences = [mf_paras_context_dict[sp_title_in_mf][i].strip() for i in sp_sentences_idx if i < len(mf_paras_context_dict[sp_title_in_mf])]
            context = [sp_title_in_mf] + sp_sentences
            context_in_prompt = "\n".join([f"- {x}" for x in context])
            return context_in_prompt
        random.shuffle(mdr_example["pos_paras"])
        start_para, bridge_para = mdr_example["pos_paras"]
        structure_info = {
            "question": normalize(mdr_example.get("question").strip()),
            "answer": normalize(mf_example.get("answer").strip()),
            "context1": get_context(start_para),
            "context2": get_context(bridge_para)
        }
        return structure_info

    def convert_info_to_prompt(self, 
                               example_info: Dict, 
                               custom_instruction: str=None) -> str:
        if custom_instruction:
            try:
                res = custom_instruction.format(**example_info)
            except:
                self.logger.info(f"Your custom instruction is illegal")
                res = self.instruction.format(**example_info)
        else:
            res = self.instruction.format(**example_info)
        return res


    def load_dataset(self):
        self.mdr_data = read_jsonl(self.config.mdr_hotpot_path)
        self.logger.info(f"Succesfully load mdr hotpot dataset, totally: {len(self.mdr_data)} examples")
        self.mf_data = read_json(self.config.mf_hotpot_path)
        self.logger.info(f"Succesfully load multifactor hotpot dataset, totally: {len(self.mf_data)} examples")

    @staticmethod
    def _correct_sp_sentences_id(sp_idxs, max_idx):
        new_idx = []
        for sp_idx in sorted(sp_idxs):
            if sp_idx - 1 >= 0 and sp_idx - 1 not in new_idx:
                new_idx.append(sp_idx - 1)
            if sp_idx not in new_idx:
                new_idx.append(sp_idx)
            if sp_idx + 1 < max_idx and sp_idx + 1 not in new_idx:
                new_idx.append(sp_idx + 1)
        return new_idx
    
    def update_processed_index(self, new_index):
        pass
        # np.save(self.processed_list_path,
        #          np.concatenate((self.processed_mdr_list, np.array(new_index)), axis=0))
        
    def check_example_is_valid(self, idx, mdr_example):
        if (self.config.stage == "summary" and self.config.summary_mode == 1) \
            or self.config.stage in "refine":
            return bool(mdr_example.get("multifactor_abs_id") is not None)
        elif (self.config.stage == "summary" and self.config.summary_mode == 2):
            if mdr_example.get("multifactor_abs_id") is not None:
                mf_sample = self.mf_data[int(mdr_example.get("multifactor_abs_id"))]
                if mf_sample.get("type") == "bridge" and mf_sample.get("full answer") != "-1":
                    return True
                else:
                    return False
            else:
                return False
        elif self.config.stage in ["comparison", "full_answer"]:
            if mdr_example.get("multifactor_abs_id") is not None and mdr_example.get("type") == "comparison":
                return True
            else:
                return False

        else:
            raise ValueError(f"check data processing mode")

            



def test_code():
    import logging
    logger = logging.getLogger(__file__)
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    file_handler = logging.FileHandler(os.path.join(r"x_retrieval/pre_processing/output/test", "run.log"))
    handlers.append(file_handler)
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    config = RawHotpotConfig()
    data = RawHotpot(config, logger)
    for x in data.get_example(2, 20):
        print(x)
    print("test")

if __name__ == "__main__":
    # demo_config = RawHotpotConfig()
    # print(demo_config)
    test_code()
    
