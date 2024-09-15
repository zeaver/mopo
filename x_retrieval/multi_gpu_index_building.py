# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pdb
import os
import time
import numpy as np
import torch
import logging
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler

from src.options import Options
from src import  slurm, dist_utils, utils, contriever, corpus_dataset


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def re_index(data_emb, data_local_idx, if_call=False):
    if if_call:
        new_data_emb = torch.zeros_like(data_emb)
        for i in range(data_local_idx.size(0)):
            new_data_emb[data_local_idx[i]] = data_emb[i]
        return new_data_emb
    else:
        return data_emb


def endoding_corpus(opt, model, tokenizer):

    corpus = corpus_dataset.HotpotCorpusDataset(
        datapaths=opt.corpus_dataset_path,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
    )
    
    # print("rank", dist_utils.get_rank())
    # print("idx", " ".join(map(str, [x["abs_idx"] for x in corpus])))
    
    collator = corpus_dataset.HotpotCollator(tokenizer, passage_maxlength=512)
    # corpus_sampler = DistributedSampler(corpus, num_replicas=dist_utils.get_world_size(), rank=dist_utils.get_rank(), shuffle=False)
    corpus_dataloader = DataLoader(
        corpus,
        # sampler=corpus_sampler,
        batch_size=256,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    model.eval()
    allemb = []
    all_idx = []
    all_doc_ids = []
    with torch.no_grad():
        for i, batch in enumerate(corpus_dataloader):
            if i <= 2000:
                batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}

                emb = model(batch["input_ids"], batch["attention_mask"])
                
                allemb.append(emb.cpu())
                all_idx.append(batch["abs_idx"].cpu())
                all_doc_ids += batch["doc_ids"]

                if i % opt.log_freq == 0:
                    log = f"{i} / {len(corpus_dataloader)} batch is encoding"
                    logger.info(log)
                if i % 500 == 0:
                    np.save(os.path.join(opt.batch_embeddings_dir, f"allemb_rank_{dist_utils.get_rank()}_batch_{i}.npy"), torch.cat(allemb, dim=0).numpy())
                    np.save(os.path.join(opt.batch_embeddings_dir, f"all_abs_id_rank_{dist_utils.get_rank()}_batch_{i}.npy"), torch.cat(all_idx, dim=0).numpy())
            else:
                break

        allemb = torch.cat(allemb, dim=0)
        allemb = allemb.cuda()
        all_idx = torch.cat(all_idx, dim=0)
        all_idx = all_idx.cuda()
        print(f"before gather queries embedding shape: {allemb.shape} in rank {dist_utils.get_rank()}")
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
            all_idx = dist_utils.varsize_gather_nograd(all_idx)
        if dist_utils.is_main():

            allemb = re_index(allemb, all_idx, False)
                
            np.save(os.path.join(opt.output_dir, "corpus_embeddings.npy"), allemb.cpu().numpy())
            np.save(os.path.join(opt.output_dir, "DDP_abs_idxs.npy"), all_idx.cpu().numpy())
            logger.info(f"saving the numpy in {opt.output_dir}")
            logger.info(f"Corpus embedding shape is {allemb.shape}")
        return all_doc_ids

def main():
    logger.info("Start")

    options = Options()
    opt = options.parse()

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    

    time_stamp = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())
    opt.output_dir = os.path.join(opt.output_dir, f"beir_hotpot_{time_stamp}")

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    os.makedirs(opt.output_dir, exist_ok=True)
    opt.batch_embeddings_dir = os.path.join(opt.output_dir, "batch_embeddings")
    os.makedirs(opt.batch_embeddings_dir, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        message = ""
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = options.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        print(message, flush=True)
        file_name = os.path.join(opt.output_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)
    
    opt.model_path = r"huggingface_download/contriever-msmarco"

    model, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_name, opt.model_path, raw_model_id=opt.raw_model_id)
    opt.retriever_model_id = retriever_model_id

    model = model.cuda()

    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["RANK"])],
            output_device=int(os.environ["RANK"]),
            find_unused_parameters=False,
        )

    logger.info("Start training")
    DDP_processing_doc_ids = endoding_corpus(opt, model, tokenizer)
    with open(os.path.join(opt.output_dir, f"rank_{dist_utils.get_rank()}_docids"), 'w') as file:
        for item in DDP_processing_doc_ids:
            file.write(str(item) + '\n')
    
    
    


if __name__ == "__main__":
    main()
