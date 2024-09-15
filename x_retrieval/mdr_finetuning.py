# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import pdb
import os
import time
from pathlib import Path
import torch
from torch.utils.tensorboard import SummaryWriter
import logging
import json
import numpy as np
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from src.options import Options
from src import mdr_finetuning_data, slurm, dist_utils, utils, contriever, mdr_inbatch
from src.eval_retrieval import EvalDataset, EvalRunner


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = logging.getLogger(__name__)


def finetuning(opt, model, optimizer, scheduler, tokenizer, step):

    run_stats = utils.WeightedAvgStats()

    tb_logger = utils.init_tb_logger(opt.output_dir)

    if hasattr(model, "module"):
        eval_model = model.module
    else:
        eval_model = model
    eval_model = eval_model.get_encoder()

    train_dataset = mdr_finetuning_data.MdrDataset(
        datapaths=opt.train_data,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        data_format=opt.data_format,
        if_add_prefix=opt.if_add_prefix,
        doc_prefix=opt.doc_prefix,
        query_prefix=opt.query_prefix,
        statement_prefix=opt.statement_prefix
    )
    if dist_utils.get_rank() == 0:
        _training_demo = train_dataset[0]
        _training_demo_str = [f"{k}: {v}" for k, v in _training_demo.items()]
        _training_demo_str = "\n" + "\n\n".join(_training_demo_str)
        logger.info(_training_demo_str)
    collator = mdr_finetuning_data.Collator(tokenizer, opt)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=True,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    # train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
    # evaluate2(opt, eval_model, tokenizer, tb_logger, step)

    epoch = 1

    model.train()
    prev_ids, prev_mask = None, None
    while step < opt.total_steps:
        logger.info(f"Start epoch {epoch}, number of batches: {len(train_dataloader)}")
        for i, batch in enumerate(train_dataloader):
            batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            step += 1

            train_loss, iter_stats = model(**batch, stats_prefix="train")
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.gradient_clip)

            if opt.optim == "sam" or opt.optim == "asam":
                optimizer.first_step(zero_grad=True)

                sam_loss, _ = model(**batch, stats_prefix="train/sam_opt")
                sam_loss.backward()
                optimizer.second_step(zero_grad=True)
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            run_stats.update(iter_stats)

            if step % opt.log_freq == 0:
                log = f"{step} / {opt.total_steps}"
                for k, v in sorted(run_stats.average_stats.items()):
                    log += f" | {k}: {v:.3f}"
                    if tb_logger:
                        tb_logger.add_scalar(k, v, step)
                log += f" | lr: {scheduler.get_last_lr()[0]:0.3g}"
                log += f" | Memory: {torch.cuda.max_memory_allocated()//1e9} GiB"

                logger.info(log)
                run_stats.reset()

            if step % opt.eval_freq == 0:

                # train.eval_model(opt, eval_model, None, tokenizer, tb_logger, step)
                evaluate2(opt, eval_model, tokenizer, tb_logger, step)

                if step % opt.save_freq == 0 and dist_utils.get_rank() == 0:
                    utils.save(
                        eval_model,
                        optimizer,
                        scheduler,
                        step,
                        opt,
                        opt.output_dir,
                        f"step-{step}",
                    )
                model.train()

            if step >= opt.total_steps:
                break
        
        if step % len(train_dataloader) == 0:
            utils.save(
                    eval_model,
                    optimizer,
                    scheduler,
                    step,
                    opt,
                    opt.output_dir,
                    f"epoch-{epoch}",
                )
        epoch += 1

@torch.no_grad()
def evaluate1(opt, model, tokenizer, tb_logger, step):
    eval_dataset = mdr_finetuning_data.MdrDataset(
        datapaths=opt.eval_data,
        normalize=opt.eval_normalize_text,
        global_rank=dist_utils.get_rank(),
        world_size=dist_utils.get_world_size(),
        maxload=opt.maxload,
        data_format=opt.data_format
    )
    collator = mdr_finetuning_data.Collator(tokenizer, passage_maxlength=opt.chunk_length, data_format=opt.data_format)
    eval_sampler = RandomSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=opt.per_gpu_batch_size,
        drop_last=False,
        num_workers=opt.num_workers,
        collate_fn=collator,
    )

    model.eval()
    if hasattr(model, "module"):
        model = model.module
    
    # all_eval_res = utils.MetricStat(["rrs_1", "rrs_2", "acc_1", "acc_2"])

    rrs_1, rrs_2, acc_1, acc_2 = [], [], [], []

    for i, batch in enumerate(eval_dataloader):
        batch = {key: value.cuda() if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        batch_eval_res = model.batch_eval(**batch)
        # all_eval_res.update(batch_eval_res)
        rrs_1.append(batch_eval_res["rrs_1"])
        rrs_2.append(batch_eval_res["rrs_2"])
        acc_1.append(batch_eval_res["acc_1"])
        acc_2.append(batch_eval_res["acc_2"])

    rrs_1 = torch.cat(rrs_1, dim=0)
    rrs_2 = torch.cat(rrs_2, dim=0)
    acc_1 = torch.cat(acc_1, dim=0)
    acc_2 = torch.cat(acc_2, dim=0)

    if dist_utils.is_main():
        mrr_1 = dist_utils.get_varsize(rrs_1).mean()
        mrr_2 = dist_utils.get_varsize(rrs_2).mean()
        acc_1 = dist_utils.get_varsize(acc_1).mean() * 100
        acc_2 = dist_utils.get_varsize(acc_2).mean() * 100

        message = []
        message = [f"eval acc_1: {acc_1:.2f}%",
                   f"eval acc_2: {acc_2:.2f}%",
                   f"eval mrr_1: {mrr_1:.3f}",
                   f"eval mrr_2: {mrr_2:.3f}",]
        logger.info(" | ".join(message))
        if tb_logger is not None:
            tb_logger.add_scalar(f"eval_acc_1", acc_1, step)
            tb_logger.add_scalar(f"eval_acc_2", acc_2, step)
            tb_logger.add_scalar(f"mrr_1", mrr_1, step)
            tb_logger.add_scalar(f"mrr_2", mrr_2, step)

def evaluate2(opt, model, tokenizer, tb_logger, step):
    return 
    if dist_utils.is_main():
        torch.cuda.empty_cache()
        eval_dataset = EvalDataset(data_path = opt.eval_dataset_path,
                                   candidates_path = opt.corpus_dataset_path,
                                   add_prefix = opt.if_add_prefix,
                                   doc_prefix=opt.doc_prefix,
                                   query_prefix=opt.query_prefix
                                   )
        eval_runner = EvalRunner(model_ckpt_or_path=(model, tokenizer),
                             eval_dataset=eval_dataset,
                             eval_args=opt,
                             device=model.device
                             )
        eval_res = eval_runner.inference(step)
        torch.cuda.empty_cache()
        logger.info(" | ".join([f"{k}: {v}" for k, v in eval_res.items()]))

def main():
    logger.info("Start")

    options = Options()
    opt = options.parse()

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.manual_seed(opt.seed)
    slurm.init_distributed_mode(opt)
    slurm.init_signal_handler()
    
    opt.output_dir = os.path.join(opt.output_dir, time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime()))

    directory_exists = os.path.isdir(opt.output_dir)
    if dist.is_initialized():
        dist.barrier()
    if dist_utils.is_main():
        Path(opt.output_dir).mkdir(parents=True, exist_ok=True)
    if not directory_exists and dist_utils.is_main():
        options.print_options(opt)
    if dist.is_initialized():
        dist.barrier()
    utils.init_logger(opt)

    step = 0

    retriever, tokenizer, retriever_model_id = contriever.load_retriever(opt.model_name, opt.model_path, raw_model_id=opt.raw_model_id)
    opt.retriever_model_id = retriever_model_id
    model = mdr_inbatch.MdrInBatch(opt, retriever, tokenizer)

    model = model.cuda()

    optimizer, scheduler = utils.set_optim(opt, model)
    # if dist_utils.is_main():
    #    utils.save(model, optimizer, scheduler, global_step, 0., opt, opt.output_dir, f"step-{0}")
    logger.info(utils.get_parameters(model))

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = opt.dropout

    if torch.distributed.is_initialized():
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[int(os.environ["RANK"])],
            output_device=int(os.environ["RANK"]),
            find_unused_parameters=False,
        )

    logger.info("Start training")
    finetuning(opt, model, optimizer, scheduler, tokenizer, step)


if __name__ == "__main__":
    main()
