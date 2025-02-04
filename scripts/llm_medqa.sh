#!/bin/bash

# for sagemaker
pip install --upgrade pip
pip install -r requirements.txt

# export CUDA_VISIBLE_DEVICES="3"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
dt=`date '+%m%d_%H%M'`


# Data Preprocessing
kg=umls
dataset="medqa"
seed=1
max_node_num=200
inhouse=false


# Others
log_interval=10
fp16=false
upcast=false
resume_checkpoint=None
resume_id=None


# Training
# lr="1e-5"
lr="5e-5"
# lr="1e-4"
# lr="2e-4"
# lr="5e-4"
lr_schedule=warmup_linear
# lr_schedule=fixed
warmup_steps=500
bs=8
n_epochs=30
max_epochs_before_stop=20


# LLM
# encoder='roberta-large'
# encoder='google/flan-t5-small'
# encoder='google/flan-t5-large'
encoder='google/flan-t5-xl'
# encoder='google/flan-t5-xxl'
max_seq_len=512
load_llm_cache=true
load_graph_cache=true


# efficient training
subsample=1.0

# baseline
baseline_flattern=false
# baseline_flattern='rs'
# baseline_flattern='bfs'


# prompt
# prompt=false
# lora=true
# prompt='regular'
prompt='gnn'


# GNN
gnn_dim=1024
gnn_layers=3

# link prediction task
link_task=0.1
link_drop_probability=0.3

# cross modality
cross_modality_layers=3


# dataset-level prompt with gnn prompt
dataset_level_prompt=false

# token numbers of 1) regular prompt tuning, and 2) dataset-level prompt
num_virtual_tokens=8

debug2=false

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "prompt: $prompt"
echo "enc_name: $encoder"
echo "batch_size: $bs"
echo "learning_rate: lr $lr"
echo "subsample: $subsample"
echo "gnn: layers $gnn_layers, dim $gnn_dim"
echo "link_task: $link_task, link_drop_probability: $link_drop_probability"
echo "cross_modality_layers: $cross_modality_layers"
echo "******************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref
mkdir -p logs




# run_name=LoRA_t5-xxl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=regular_prompt_t5-xxl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=t5-xl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=t5-xxl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=gat_linkTask${link_task}_LinkDrop${link_drop_probability}_t5-xxl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=sample${subsample}_regular_prompt_t5-xl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=sample${subsample}_gat${gnn_layers}_linkTask${link_task}_croMod${cross_modality_layers}_t5-xl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
run_name=sample${subsample}_P${prompt}${gnn_layers}_warm${warmup_steps}_linkTask${link_task}_croMod${cross_modality_layers}_t5-xl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=sample${subsample}_P${prompt}_t5-xxl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}


###### Training ######
python3 -u llm.py \
    --dataset $dataset \
    --encoder $encoder --gnn_dim $gnn_dim -lr $lr -bs $bs --seed $seed -sl ${max_seq_len} --max_node_num ${max_node_num} \
    --n_epochs $n_epochs --max_epochs_before_stop ${max_epochs_before_stop} --fp16 $fp16 --upcast $upcast \
    --save_dir ${save_dir_pref}/${dataset}/${run_name} --save_model 0 \
    --run_name ${run_name} \
    --resume_checkpoint ${resume_checkpoint} --resume_id ${resume_id} --lr_schedule ${lr_schedule} --warmup_steps $warmup_steps -ih ${inhouse} --kg $kg \
    --load_llm_cache ${load_llm_cache} \
    --load_graph_cache ${load_graph_cache} \
    --log_interval ${log_interval} \
    --prompt ${prompt} \
    --cross_modality_layers ${cross_modality_layers} \
    --link_task ${link_task} \
    --link_drop_probability ${link_drop_probability} \
    --dataset_level_prompt ${dataset_level_prompt} \
    --num_virtual_tokens ${num_virtual_tokens} \
    --gnn_layers ${gnn_layers} \
    --subsample ${subsample} \
    --baseline_flattern ${baseline_flattern} \
    --debug2 ${debug2} \
    # --mode eval \
