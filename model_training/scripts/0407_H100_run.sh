

# export CUDA_VISIBLE_DEVICES=0
set -ex

cd 

combine_weights=("1.0" "0.0" 1e-3 1e-6)
[ -z "$SLURM_ARRAY_TASK_ID" ] && SLURM_ARRAY_TASK_ID=1
export COMBINE_WEIGHT=${combine_weights[$SLURM_ARRAY_TASK_ID - 1]}

export DATE=0407
export NODE_TYPE="H100"
scprint_spatial fit --config config/0303_base_spatial_all_crossattention.yml --scprint_training.name ${DATE}_base_spatial_all_crossattention_weight${COMBINE_WEIGHT}_${NODE_TYPE} --model.combine_weight ${COMBINE_WEIGHT} --data.batch_size 128
