#!/bin/bash
rm -rf ./logs/runs/*
export XDG_CACHE_HOME=/lscratch/$SLURM_JOB_ID
checkpoint_dir=/data/saveryme/trec_2020/models/hierarchical_mds/checkpoints
prediction_dir=/data/saveryme/trec_2020/models/hierarchical_mds/predictions
log_dir=/data/saveryme/trec_2020/models/hierarchical_mds/logs/runs
data_dir=/data/LHC_kitchensink/tensorflow_datasets_max/downloads/manual
max_seq_len=256
max_docs=5
epochs=100
batch_size=16
hidden_dim=256
ffw_dim=1024
att_heads=4
# Valid tasks: 'mediqa', 'bioasq', 'ebm', 'medlineplus', 'cnn_dailymail', 'eli5'
#cnn_dailymail eli5
python -u -m hier_mds.run_train \
  --train \
  --training_tasks="mediqa bioasq" \
  --validate \
  --validation_tasks="mediqa" \
  --hf_model="facebook/bart-large" \
  --data_dir=$data_dir \
  --cache_dir=/lscratch/$SLURM_JOB_ID \
  --checkpoint_dir=$checkpoint_dir \
  --prediction_dir=${pred_dir} \
  --init_checkpoint=None \
  --seed=42 \
  --max_docs=$max_docs \
  --enc_hidden_dim=$hidden_dim \
  --dec_hidden_dim=$hidden_dim \
  --ffw_dim=$ffw_dim \
  --max_seq_len=$max_seq_len \
  --epochs=${epochs} \
  --batch_size=$batch_size \
  --eval_batches=10 \
  --local_enc_layers=2 \
  --global_enc_layers=2 \
  --dec_layers=2 \
  --n_att_heads=$att_heads \
  --decoder_attn_mask \
  --dec_mem_input="local" \
  --tensorboard \
  --log_dir=$log_dir \
  --init_bert_weights \
  --multi_head_pooling \
  --query_doc_attn \
  --doc_mixing \
  --mmr \
  #--label_smoothing=.1 \
