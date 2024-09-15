We provide code of MoPo and a small sample of PostSumQA here.
The datasets and weights used for evaluation are provided in the supplementary materials of the paper.
## MoPo
To run the retrieval test in the paper, please run the following command:
**prepare the environment**
```
conda create -n mopo python=3.9
conda activate mopo
pip install -r requirements.txt
```


**download and train related models**
1. download e5-base-v2 and flan-t5-large models
   ```
   sh download/e5_base_v2/download.sh
   sh download/flan_t5_large/download.sh
   ```
2. train summarizer model flan-t5-large
   ```
   python sum_output/training_t5/training_xl.py
   ```
   **Note**: we provide a checkpoint for e5-base-v2 in 'x_retrieval/experiments/mopo/checkpoint.pth', so we don't need to train it again. 

**run retrieval test**
```
CUDA_VISIBLE_DEVICES=0 python x_retrieval/beir_eval.py --model_name e5 --model_path x_retrieval/experiments/mopo --if_add_prefix 1 --doc_prefix passage --query_prefix query --norm_query 1 --norm_doc 1 --faiss_gpu
```
Here will print the 'eval_output_dir' path in the terminal. Then replace the '\$eval_output_dir\$' with the printed path in the following command and run it to get the final results:
```
CUDA_VISIBLE_DEVICES=0 python x_retrieval/src/summarizer_modeling.py --Result_path $eval_output_dir$ --Save_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo

python -m torch.distributed.launch --nproc_per_node=4 sum_output/infra_t5/run_infra.py --Input_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo

CUDA_VISIBLE_DEVICES=0 python final_eval.py --model_path x_retrieval/experiments/mopo --hop1_res_path $eval_output_dir$ --hop2_quries_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo --faiss_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo
```

## PostSumQA

We provide a small sample of PostSumQA including 50 samples of PostSumQA in PostSumQA_sample.jsonl

## Based on Previous Work

This code has been developed based on the foundational work of the following predecessors:
- [MDR](https://github.com/facebookresearch/multihop_dense_retrieval)
- [Contriever](https://github.com/facebookresearch/contriever)
