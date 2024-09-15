'''
- https://medium.com/@geronimo7/llms-multi-gpu-inference-with-accelerate-5a8333e4c5db
'''

from accelerate import Accelerator
from accelerate.utils import gather_object
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from statistics import mean
import torch, time, json
import argparse

from typing import Any

# utiles
#########################################################################################
from pathlib import Path
from tqdm import tqdm
from copy import deepcopy

parser = argparse.ArgumentParser()
parser.add_argument("--Input_path", default='', type=str, help="input path")


args = parser.parse_args()

hop2_input_path = args.Input_path


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

#########################################################################################

accelerator = Accelerator()

# load prompts

prompts_raw = read_jsonl(hop2_input_path+"/t5_hop2_inputs.jsonl")
prompts_all = [x.get('hop2_inst') for x in prompts_raw]

# print demo
_demos = prompts_all[:4]
print("=======================================================================\n")
print("\n\n".join([f"Demo#{i}:\n" + _demos[i] for i in range(len(_demos))]))
print("\n=======================================================================")

# load a base model and tokenizer
model_path = "x_retrieval/results/t5_res_xl/final_save"
raw_model_path = "download/flan_t5_large"
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_path,    
    device_map={"": accelerator.process_index},
    torch_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(raw_model_path)   

def prepare_prompts(prompts, tokenizer, batch_size=32):
    batches=[prompts[i:i + batch_size] for i in range(0, len(prompts), batch_size)]  
    batches_tok=[]
    for prompt_batch in batches:
        batch_text = [x.get('hop2_inst') for x in prompt_batch]
        batches_tok.append(
            tokenizer(
                batch_text, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                pad_to_multiple_of=8,
                max_length=400).to("cuda") 
            )

    return batches_tok, batches

# sync GPUs and start the timer
accelerator.wait_for_everyone()    
start=time.time()
print("start")
# divide the prompt list onto the available GPUs
with accelerator.split_between_processes(prompts_raw) as prompts:
    results=dict(outputs=[], num_tokens=0, raw_info=[])

    # have each GPU do inference in batches
    prompt_batches_toks, prompt_batches = prepare_prompts(prompts, tokenizer, batch_size=32)

    for prompts_tokenized, prompt_raw in zip(prompt_batches_toks, prompt_batches):
        outputs_tokenized=model.generate(**prompts_tokenized, max_new_tokens=60)

        # # remove prompt from gen. tokens
        # outputs_tokenized=[ tok_out[len(tok_in):]
        #     for tok_in, tok_out in zip(prompts_tokenized["input_ids"], outputs_tokenized) ]

        # count and decode gen. tokens
        num_tokens=sum([ len(t) for t in outputs_tokenized ])
        outputs = tokenizer.batch_decode(
            outputs_tokenized,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # store in results{} to be gathered by accelerate
        results["outputs"].extend(outputs)
        results["num_tokens"] += num_tokens
        results["raw_info"].extend(prompt_raw)


    results=[ results ] # transform to list, otherwise gather_object() will not collect correctly

# collect results from all the GPUs
results_gathered=gather_object(results)

if accelerator.is_main_process:
    timediff=time.time()-start
    num_tokens=sum([r["num_tokens"] for r in results_gathered ])

    print(f"tokens/sec: {num_tokens//timediff}, time elapsed: {timediff}, num_tokens {num_tokens}")

    outs = [x for r in results_gathered for x in r["outputs"]]
    raws = [x for r in results_gathered for x in r["raw_info"]]
    print(outs[1])

    with open(hop2_input_path+"/t5_hop2_outputs.jsonl" , "w") as f:
        for o, r in zip(outs, raws):
            r.update({"hop2_generated_query": o})
            f.write(json.dumps(r) + '\n')
