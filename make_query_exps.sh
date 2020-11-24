#!/bin/bash
rm query_exps.swarm
data_dir=/data/LHC_kitchensink/tensorflow_datasets_max/downloads/manual
max_seq_len=256
max_docs=5
epochs=200
batch_size=16
hidden_dim=256
ffw_dim=1024
att_heads=4
decoder_input="local"

# Full sink except medlineplus_reviews
training_tasks="mediqa bioasq"
validation_tasks="mediqa bioasq"
for query_attn in query_doc_attn mmr
do
    exp_name="query_attn=${query_attn}-dec_input=$decoder_input"
    checkpoint_dir=/data/saveryme/trec_2020/models/hierarchical_mds/checkpoints/$exp_name 
    prediction_dir=/data/saveryme/trec_2020/models/hierarchical_mds/predictions/$exp_name
    log_dir=/data/saveryme/trec_2020/models/hierarchical_mds/logs/runs/$exp_name
    echo "echo $exp_name; export XDG_CACHE_HOME=/lscratch/\$SLURM_JOB_ID; mkdir -p ${prediction_dir}; mkdir -p ${log_dir}; \\
    /data/saveryme/conda/envs/hier_mds_env/bin/python -u -m hier_mds.run_train \\
      --train \\
      --training_tasks="\"mediqa bioasq\"" \\
      --validate \\
      --validation_tasks=\""mediqa\"" \\
      --hf_model="\"facebook/bart-large"\" \\
      --data_dir=$data_dir \\
      --cache_dir=/lscratch/\$SLURM_JOB_ID \\
      --checkpoint_dir=$checkpoint_dir \\
      --prediction_dir=$pred_dir \\
      --init_checkpoint=None \\
      --seed=42 \\
      --max_docs=$max_docs \\
      --enc_hidden_dim=$hidden_dim \\
      --dec_hidden_dim=$hidden_dim \\
      --ffw_dim=$ffw_dim \\
      --max_seq_len=$max_seq_len \\
      --epochs=$epochs \\
      --batch_size=$batch_size \\
      --eval_batches=10 \\
      --local_enc_layers=2 \\
      --global_enc_layers=2 \\
      --dec_layers=2 \\
      --n_att_heads=$att_heads \\
      --decoder_attn_mask \\
      --dec_mem_input="\"$decoder_input\"" \\
      --tensorboard \\
      --log_dir=$log_dir \\
      --init_bert_weights \\
      --multi_head_pooling \\
      --doc_mixing \\
      --$query_attn" >> query_exps.swarm
((i=i+1))
done

