#!/bin/bash
export XDG_CACHE_HOME=/lscratch/$SLURM_JOB_ID
checkpoint_dir=/data/saveryme/trec_2020/models/hierarchical_mds/checkpoints
prediction_dir=/data/saveryme/trec_2020/models/hierarchical_mds/predictions
max_seq_len=256
max_docs=5
epochs=1
batch_size=3
hidden_dim=128
ffw_dim=1024
att_heads=4
# Valid tasks: 'mediqa', 'bioasq', 'ebm', 'medlineplus', 'cnn_dailymail', 'eli5'
python -u -m hier_mds.run_train \
  --training_tasks="ebm" \
  --validation_tasks="ebm" \
  --hf_model="facebook/bart-large" \
  --data_dir=/data/LHC_kitchensink/tensorflow_datasets_max/downloads/manual \
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
  --label_smoothing=.1 \
  --dec_layers=2 \
  --n_att_heads=$att_heads \
  --decoder_attn_mask \
  --multi_head_pooling \
  --mmr \
  --doc_mixing \
  --padding_mask \
  --train \
  --validate

#--query_doc_attn \
#--init_bert_weights
