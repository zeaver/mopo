#python -m torch.distributed.launch --nproc_per_node=4 /x_retrieval/mdr_finetuning.py --model_name e5 --model_path /download/e5_base_v2  --if_add_prefix 1 --doc_prefix passage --query_prefix query --statement_prefix query --norm_query 1 --norm_doc 1 --output_dir /x_retrieval/experiments/train_e5v2_mdr_proposal --data_format mix --if_load_teacher_retriever 0 --temperature 0.01 --momentum 0.99

CUDA_VISIBLE_DEVICES=0 python x_retrieval/beir_eval.py --model_name e5 --model_path x_retrieval/experiments/mopo --if_add_prefix 1 --doc_prefix passage --query_prefix query --norm_query 1 --norm_doc 1 --faiss_gpu

CUDA_VISIBLE_DEVICES=0 python x_retrieval/src/summarizer_modeling.py --Result_path $eval_output_dir$ --Save_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo

python -m torch.distributed.launch --nproc_per_node=4 sum_output/infra_t5/run_infra.py --Input_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo

CUDA_VISIBLE_DEVICES=0 python final_eval.py --model_path x_retrieval/experiments/mopo --hop1_res_path $eval_output_dir$ --hop2_quries_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo --faiss_path x_retrieval/experiments/beir_eval_output/beir_hotpot/mopo