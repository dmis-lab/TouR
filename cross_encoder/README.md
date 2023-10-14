# Train phrase re-rankers

First, please download the training/dev set.

- [multi.train.v4.1.json](https://drive.google.com/file/d/1E6Gu5dCbTFrfy5mp5wX2G-U29f1RHkwD/view?usp=sharing): Phrase-ranker training set consisting of data from multiple sources (NQ|TriviaQA|SQuAD|WQ|TREC)
- [nq.dev.v4.1.json](https://drive.google.com/file/d/1E6OrUKwVDIPzco7CjwPA43o6iEHyPPOa/view?usp=sharing): Phrase-ranker dev set consisting of data from NQ

Second, please run the following command.
```bash
# install scikit-learn
$ pip install scikit-learn

# train phrase-reranker-multi
$ python run_labeler.py \
    --model_name_or_path dmis-lab/roberta-large-nqtqatrecwqsqd-mrc   \
    --train_file multi.train.v4.1.json \
    --validation_file nq.dev.v4.1.json \
    --do_train   \
    --do_eval   \
    --pad_to_max_length False\
    --max_seq_length 512   \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5   \
    --num_train_epochs 1   \
    --fp16 \
    --load_best_model_at_end \
    --metric_for_best_model accuracy \
    --evaluation_strategy steps \
    --save_steps 11000 \
    --eval_steps 11000 \
    --logging_steps 11000 \
    --save_total_limit 1 \
    --output_dir phrase-reranker-multi \
    --overwrite_output_dir
```