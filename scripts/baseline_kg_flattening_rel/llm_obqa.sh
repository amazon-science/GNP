#!/bin/bash

# for sagemaker
pip install --upgrade pip
pip install -r requirements.txt

# export CUDA_VISIBLE_DEVICES="0,1,2"
# export CUDA_VISIBLE_DEVICES="0,1,2,3"
# export CUDA_VISIBLE_DEVICES="4,5,6,7"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=4
dt=`date '+%m%d_%H%M'`


# Data Preprocessing
kg=cpnet
dataset="obqa"
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
# lr="5e-5"
lr="1e-4"
# lr="2e-4"
lr_schedule=warmup_linear
# lr_schedule=fixed
warmup_steps=500
bs=8
n_epochs=50
max_epochs_before_stop=20



# LLM
# encoder='roberta-large'
# encoder='google/flan-t5-small'
# encoder='google/flan-t5-large'
encoder='google/flan-t5-xl'
# encoder='google/flan-t5-xxl'
max_seq_len=100
load_llm_cache=true
load_graph_cache=true
# load_llm_cache=false
# load_graph_cache=false

# efficient training
subsample=1.0

# baseline
# baseline_flattern=false
baseline_flattern='rs'
# baseline_flattern='bfs'
# baseline_flattern='every_triplet'
# baseline_flattern='triplet_qa'
# baseline_flattern='triplet_q'
# baseline_flattern='triplet_a'


# prompt
prompt=false
# prompt='regular'
# prompt='gnn'


# prompt design
prompt_design=false
# prompt_design='prompt_A'
# prompt_design='prompt_B'
# prompt_design='prompt_C'

# GNN
gnn_dim=1024
use_relational_gnn=false
gnn_layers=3
link_task=0.1
link_drop_probability=0.3
cross_modality_layers=1
moe_experts=0
moe_loss_weight=0


# lora
lora=false

# dataset-level prompt with gnn prompt
dataset_level_prompt=false
# token numbers of 1) regular prompt tuning, and 2) dataset-level prompt
num_virtual_tokens=8

# debug
debug2=false

# ablation study
no_projector=false

# case study for visualization
case_study=false

# save model wrong prediction for case study
save_model_wrong_prediction=false


# statements
if [ $lora == true ] 
then
    echo "lora: true"
    n_epochs=10
    if [ "$prompt" = 'gnn' ]
    then
        n_epochs=10
    fi
fi

# xxl parameters
if [ "$encoder" = 'google/flan-t5-xxl' ]
then
    cross_modality_layers=3
fi

echo "***** hyperparameters *****"
echo "dataset: $dataset"
echo "n_epochs: $n_epochs"
echo "prompt: $prompt"
echo "enc_name: $encoder"
echo "batch_size: $bs"
echo "learning_rate: lr $lr"
echo "subsample: $subsample"
echo "gnn: layers $gnn_layers, dim $gnn_dim"
echo "link_task: $link_task, link_drop_probability: $link_drop_probability"
echo "cross_modality_layers: $cross_modality_layers"
echo "lora: $lora"
echo "use_relational_gnn: $use_relational_gnn"
echo "moe_experts: $moe_experts"
echo "moe_loss_weight: $moe_loss_weight"
echo "no_projector: $no_projector"
echo "******************************"

save_dir_pref='runs'
mkdir -p $save_dir_pref
mkdir -p logs


run_name=${dataset}_lora${lora}_P${prompt}${gnn_layers}_rel${use_relational_gnn}_moe${moe_experts}loss${moe_loss_weight}_warm${warmup_steps}_linkTask${link_task}_${link_drop_probability}_croMod${cross_modality_layers}_t5-xl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=${dataset}_lora${lora}_P${prompt}${gnn_layers}_rel${use_relational_gnn}_moe${moe_experts}loss${moe_loss_weight}_warm${warmup_steps}_linkTask${link_task}_${link_drop_probability}_croMod${cross_modality_layers}_t5-xxl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=${dataset}_lora${lora}_P${prompt}_warm${warmup_steps}_t5-xl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=${dataset}_lora${lora}_P${prompt}_warm${warmup_steps}_t5-xxl_${dataset}_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=ablation_${dataset}_croMod${cross_modality_layers}_linkTask${link_task}_noProj${no_projector}_dataPrompt${dataset_level_prompt}_P${prompt}${gnn_layers}_t5-xl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=ablation_${dataset}_croMod${cross_modality_layers}_linkTask${link_task}_noProj${no_projector}_dataPrompt${dataset_level_prompt}_P${prompt}${gnn_layers}_t5-xxl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=paramSens_${dataset}_gnnLayers${gnn_layers}_t5-xl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=paramSens_${dataset}_gnnLayers${gnn_layers}_t5-xxl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=paramSens_${dataset}_croMod${cross_modality_layers}_t5-xl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=paramSens_${dataset}_croMod${cross_modality_layers}_t5-xxl_lr${lr}_b${bs}_e${n_epochs}_date${dt}
# run_name=baseline_${baseline_flattern}_${dataset}_xxl

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
    --prompt_design ${prompt_design} \
    --lora ${lora} \
    --use_relational_gnn ${use_relational_gnn} \
    --moe_experts ${moe_experts} \
    --moe_loss_weight ${moe_loss_weight} \
    --no_projector ${no_projector} \
    --case_study ${case_study} \
    --save_model_wrong_prediction ${save_model_wrong_prediction} \
    --mode eval \


# > ${log}
# --use_cross_modality_for_gnn ${use_cross_modality_for_gnn} \
# --use_cross_modality_for_text ${use_cross_modality_for_text} \
# --cross_modality_for_text_layers ${cross_modality_for_text_layers} \
# --use_wandb false \
# --data_dir data \
# --ent_emb_paths ${ent_emb//,/ } --kg_vocab_path $kg_vocab_path 
# --ie_dim ${ie_dim} --info_exchange ${info_exchange} --ie_layer_num ${ie_layer_num} --sep_ie_layers ${sep_ie_layers} --random_ent_emb ${random_ent_emb} 
# --load_model_path $load_model_path \
# -mbs ${mbs} 

# use_cross_modality_for_text=false
# cross_modality_for_text_layers=3
# ent_emb=data/cpnet/tzw.ent.npy
# kg_vocab_path=data/cpnet/concept.txt
# log=logs/train__${run_name}.log.txt
