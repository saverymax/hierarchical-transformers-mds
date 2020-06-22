#!/bin/bash

checkpoint_dir=/data/saveryme/trec_2020/models/hierarchical_mds/checkpoints
prediction_dir=/data/saveryme/trec_2020/models/hierarchical_mds/predictions
max_seq_len=4
max_docs=3
epochs=2
batch_size=2
hidden_dim=6
att_heads=2
python -u -m model.run_train \
  --checkpoint_dir=$checkpoint_dir \
  --data_path=/data/LHC_kitchensink/tensorflow_datasets_max \
  --prediction_dir=${pred_dir} \
  --init_checkpoint=None \
  --seed=42 \
  --vocab_size=30000 \
  --max_docs=$max_docs \
  --enc_hidden_dim=$hidden_dim \
  --dec_hidden_dim=$hidden_dim \
  --max_seq_len=$max_seq_len \
  --epochs=${epochs} \
  --batch_size=$batch_size \
  --eval_batch_size=8 \
  --eval_batches=10 \
  --local_enc_layers=4 \
  --global_enc_layers=2 \
  --dec_layers=2 \
  --n_att_heads=$att_heads \
  --top_k=2 \
  --multi_head_pooling \
  --train \

#--init_bert_weights
