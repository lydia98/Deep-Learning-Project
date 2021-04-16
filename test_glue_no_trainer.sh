python3 run_glue_no_trainer.py \
--model_name_or_path bert-base-cased \
--train_file train_df.csv \
--validation_file dev_df.csv \
--max_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir ./tmp/
