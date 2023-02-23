# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=0
dataset=Taskdata

basepath=/mnt/nfs-storage/data
modelcate=base
# modelcate=large

datapath=data/$dataset
tokpath=${basepath}/pretrained-model/bart-$modelcate
MODEL=$1
interval=1

lr=1e-5

outpath=output/${dataset}-bart-$modelcate-textinf-${lr}-DAPT

mkdir -p $outpath

python -u -m torch.distributed.launch --nproc_per_node=1 --master_port 86887 run_DAPT.py \
  --train_file $datapath/train.txt \
  --val_file $datapath/val.txt \
  --output_dir $outpath \
  --mlm_text \
  --block_size 512 \
  --per_gpu_train_batch_size 4 \
  --gradient_accumulation_steps 1  \
  --model_type "facebook/bart-$modelcate" \
  --model_name_or_path $MODEL \
  --tokenizer_name_or_path $tokpath \
  --save_total_limit 2 \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --num_train_epochs 100  \
  --learning_rate $lr \
  --joint_train_interval $interval \
  --warmup_steps 2500 \
  --max_steps 100000 \
  --logging_steps 1000 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
