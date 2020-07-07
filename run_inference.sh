#!/bin/bash

checkpoint_dir=/data/saveryme/trec_2020/models/hierarchical_mds/checkpoints
prediction_dir=/data/saveryme/trec_2020/models/hierarchical_mds/predictions
max_seq_len=4
max_docs=3
epochs=2
batch_size=2
hidden_dim=128
att_heads=2
python -u -m hier_mds.run_inference \
  --checkpoint_dir=$checkpoint_dir \
  --data_path=/data/LHC_kitchensink/tensorflow_datasets_max \
  --prediction_dir=${pred_dir} \
  --init_checkpoint=None \
  --seed=42 \
  --batch_size=$batch_size \
  --test \

  #--mmr \
#--init_bert_weights
