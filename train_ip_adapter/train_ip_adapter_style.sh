#!/usr/bin/env bash
export ENCODER_DIR='../ip_adapter_models/models/image_encoder'
export PRETRAINED_IP_DIR='../ip_adapter_models/models/ip-adapter_sd15.bin'
# export PRETRAINED_IP_DIR='../ip_adapter_models/models/ip-adapter-plus_sd15.bin'
export OUTPUT_DIR='../ip_adapter_models/indoor_raw_output'
export DATASET_FILE='training_data.json'
export DATASET_ROOT='training_images'

accelerate launch train_ip_adapter_style.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --pretrained_ip_adapter_path="$PRETRAINED_IP_DIR" \
  --image_encoder_path="$ENCODER_DIR" \
  --data_json_file="$DATASET_FILE" \
  --data_root_path="$DATASET_ROOT" \
  --mixed_precision="fp16" \
  --resolution=512 \
  --train_batch_size=8 \
  --dataloader_num_workers=4 \
  --learning_rate=1e-04 \
  --weight_decay=0.01 \
  --output_dir="$OUTPUT_DIR" \
  --save_steps=500 \