test_path='/hdd1/mujeen/DensePhrases/densephrases-data/open-qa/nq-open/nq_test_preprocessed.json'
load_dir='/hdd1/mujeen/retrieval_prf/densephrases-multi-query-nq'
ce_dir='/hdd1/mujeen/retrieval_prf/output/labeler_multi.train.v4.1_nq.dev.v4.1_rlmulti_title'
output_dir='/hdd1/mujeen/retrieval_prf/output/multi_emnlp_topk_hard'
pred_path='pred/nq_test_preprocessed_3610_tour.pred'

# --dump_dir /nvme1/jinhyuk/outputs/densephrases-multi_wiki-dev/dump \
# --index_name start/16384_flat_OPQ96 \
# --draft

CUDA_VISIBLE_DEVICES=0 python -u run_tour_densephrases.py \
	--run_mode test_query_vec \
	--cache_dir /hdd1/mujeen/DensePhrases/cache \
	--test_path ${test_path} \
	--per_device_train_batch_size 1 \
	--warmup_steps 0 \
	--dump_dir /nvme1/jinhyuk/outputs/densephrases-multi_wiki-20181220/dump \
	--index_name start/1048576_flat_OPQ96 \
	--load_dir ${load_dir} \
	--output_dir ${output_dir} \
	--pseudo_labeler_name_or_path ${ce_dir} \
	--pseudo_labeler_type top_p_hard \
	--pseudo_labeler_p 0.5 \
	--pseudo_labeler_temp 0.5 \
	--learning_rate 1.2 \
	--num_train_epochs 3 \
	--top_k 10 \
	--cuda \
	--truecase \
    --top1_earlystop \
	--truecase_path /hdd1/mujeen/DensePhrases/densephrases-data/truecase/english_with_questions.dist
