# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import argparse
import os


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize()

    def initialize(self):

        # input and output
        self.parser.add_argument("--output_dir", type=str, default="x_retrieval/experiments/test", help="models are saved here")
        self.parser.add_argument(
            "--train_data",
            nargs="+",
            default=["x_retrieval/datasets/mdr_hotpot/train.jsonl"],
            help="Data used for training, passed as a list of directories splitted into tensor files.",
        )

        ## model config
        self.parser.add_argument("--model_name", type=str, default="e5", choices=["contriever", "facebook_mdr", "e5"])
        self.parser.add_argument("--model_path", type=str, default="download/e5_base_v2", help="path for retraining")
        self.parser.add_argument("--raw_model_id", type=str, default="download/e5_base_v2", help="path for retraining")

        # training hyper-params
        self.parser.add_argument("--per_gpu_batch_size", default=16, type=int, help="Batch size per GPU for training.")
        self.parser.add_argument("--per_gpu_eval_batch_size", default=64, type=int, help="Batch size per GPU for evaluation." )
        self.parser.add_argument("--gck_segment", type=int, default=1)
        self.parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
        self.parser.add_argument("--total_steps", type=int, default=400)
        self.parser.add_argument("--warmup_steps", type=int, default=20)
        self.parser.add_argument("--chunk_length", type=int, default=350)
        self.parser.add_argument("--query_length", type=int, default=100)
        self.parser.add_argument("--passage_length", type=int, default=350)

        # logger
        self.parser.add_argument("--log_freq", type=int, default=1, help="log train stats every <log_freq> steps during training")
        self.parser.add_argument("--eval_freq", type=int, default=200, help="evaluate model every <eval_freq> steps during training")
        self.parser.add_argument("--save_freq", type=int, default=200, help="save model every <eval_freq> steps during training")

        # data format
        self.parser.add_argument("--data_format", type=str, default="mix")
        self.parser.add_argument("--if_add_prefix", type=int, default=1, choices=[0,1])
        self.parser.add_argument("--doc_prefix", type=str, default="search_document")
        # self.parser.add_argument("--doc_prefix", type=str, default="document")
        self.parser.add_argument("--query_prefix", type=str, default="search_query")
        # self.parser.add_argument("--query_prefix", type=str, default="query")
        self.parser.add_argument("--statement_prefix", type=str, default="query")
        self.parser.add_argument("--norm_query", type=int, default=0, choices=[0,1])
        self.parser.add_argument("--norm_doc", type=int, default=0, choices=[0,1])

        # knowledge distillation
        self.parser.add_argument("--if_load_teacher_retriever", type=int, default=0, choices=[0,1])
        self.parser.add_argument("--teacher_abaltion", type=bool, default=False)
        self.parser.add_argument("--teacher_optimization_mode", type=str, default="moco")
        self.parser.add_argument("--teacher_retriever_id", type=str, default="x_retrieval/experiments/mdr_teacher/2024_03_20_10_39_05/checkpoint/step-200")
        self.parser.add_argument("--teacher_loss_weight", type=float, default=1.0)
        self.parser.add_argument("--momentum", type=float, default=0.9)

        # evaluation
        self.parser.add_argument("--corpus_dataset_path", type=str, default="x_retrieval/datasets/mdr_hotpot/mdr_dev_corpus_neg20.jsonl")
        self.parser.add_argument("--eval_dataset_path", type=str, default="x_retrieval/datasets/mdr_hotpot/mdr_dev_neg20.jsonl")
        self.parser.add_argument("--hop1_res_path", type=str, default="/x_retrieval_res/hop1_res")

        self.parser.add_argument(
            "--eval_data",
            nargs="+",
            default=[],
            help="Data used for evaluation during finetuning, this option is not used during contrastive pre-training.",
        )
        self.parser.add_argument(
            "--eval_datasets", nargs="+", default=["beir_hotpot"], help="List of datasets used for evaluation, in BEIR format"
        )
        self.parser.add_argument(
            "--eval_datasets_dir", type=str, default="x_retrieval/datasets/", help="Directory where eval datasets are stored"
        )
        
        self.parser.add_argument("--continue_training", action="store_true")
        self.parser.add_argument("--faiss_gpu", action="store_true")
        self.parser.add_argument("--num_workers", type=int, default=4)

        
        self.parser.add_argument("--loading_mode", type=str, default="split")
        self.parser.add_argument("--lower_case", action="store_true", help="perform evaluation after lowercasing")
        self.parser.add_argument(
            "--sampling_coefficient",
            type=float,
            default=0.0,
            help="coefficient used for sampling between different datasets during training, \
                by default sampling is uniform over datasets",
        )
        self.parser.add_argument("--augmentation", type=str, default="none")
        self.parser.add_argument("--prob_augmentation", type=float, default=0.0)

        self.parser.add_argument("--dropout", type=float, default=0.1)
        self.parser.add_argument("--rho", type=float, default=0.05)

        self.parser.add_argument("--contrastive_mode", type=str, default="moco")
        self.parser.add_argument("--queue_size", type=int, default=65536)
        self.parser.add_argument("--temperature", type=float, default=0.01)
        
        self.parser.add_argument("--moco_train_mode_encoder_k", action="store_true")
        self.parser.add_argument("--eval_normalize_text", action="store_true")
        self.parser.add_argument("--projection_size", type=int, default=768)

        self.parser.add_argument("--ratio_min", type=float, default=0.1)
        self.parser.add_argument("--ratio_max", type=float, default=0.5)
        self.parser.add_argument("--score_function", type=str, default="dot")
        self.parser.add_argument("--retriever_model_id", type=str, default="bert-base-uncased")
        self.parser.add_argument("--pooling", type=str, default="average")
        self.parser.add_argument("--random_init", action="store_true", help="init model with random weights")
        
        self.parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
        self.parser.add_argument("--main_port", type=int, default=-1, help="Master port (for multi-node SLURM jobs)")
        self.parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
        # training parameters
        self.parser.add_argument("--optim", type=str, default="adamw")
        self.parser.add_argument("--scheduler", type=str, default="linear")
        
        self.parser.add_argument(
            "--lr_min_ratio",
            type=float,
            default=0.0,
            help="minimum learning rate at the end of the optimization schedule as a ratio of the learning rate",
        )
        self.parser.add_argument("--weight_decay", type=float, default=0.01, help="learning rate")
        self.parser.add_argument("--beta1", type=float, default=0.9, help="beta1")
        self.parser.add_argument("--beta2", type=float, default=0.999, help="beta2")
        self.parser.add_argument("--eps", type=float, default=1e-6, help="eps")
        self.parser.add_argument("--gradient_clip", type=float, default=1.0, help="clip")
        
        self.parser.add_argument("--maxload", type=int, default=None)
        self.parser.add_argument("--label_smoothing", type=float, default=0.0)

        # finetuning options
        self.parser.add_argument("--negative_ctxs", type=int, default=1)
        self.parser.add_argument("--negative_hard_min_idx", type=int, default=0)
        self.parser.add_argument("--negative_hard_ratio", type=float, default=0.0)
        
        # eval options
        #self.parser.add_argument("--eval_output_dir", type=str, default="/home/student2021//x_retrieval/results")
        #self.parser.add_argument("--corpus_dataset_path", type=str, default="/home/student2021//x_retrieval/datasets/beir_hotpot/corpus.jsonl")

    def print_options(self, opt):
        message = ""
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = f"\t[default: %s]" % str(default)
            message += f"{str(k):>40}: {str(v):<40}{comment}\n"
        print(message, flush=True)
        model_dir = os.path.join(opt.output_dir, "models")
        if not os.path.exists(model_dir):
            os.makedirs(os.path.join(opt.output_dir, "models"))
        file_name = os.path.join(opt.output_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self):
        opt, _ = self.parser.parse_known_args()
        # opt = self.parser.parse_args()
        return opt
