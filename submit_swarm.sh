# Query experiments
swarm_file=query_exps.swarm
slurm_logs=slurm_logs/query_exps
slurm_log_dir=/data/saveryme/trec_2020/models/hierarchical_mds/logs/$slurm_logs

# Document sampleing experiments
# Here lies stuff

mkdir -p $slurm_log_dir

swarm -f $swarm_file -g 64 --logdir=$slurm_log_dir --time=12:00:00 --gres=lscratch:128,gpu:v100x:1 --partition=gpu --sbatch "--ntasks=8"
