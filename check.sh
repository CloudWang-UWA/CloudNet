mkdir -p docs/inventory/experiment_trace

for f in image_generation.sh image_generation_style.sh image_generation.py image_generation_style.py logs/image_gen/*.err logs/image_gen/*.out; do
  if [ -e "$f" ]; then
    stat -c '%y  %n' "$f"
  fi
done > docs/inventory/experiment_trace/image_gen_file_times.txt

grep -R \
  -e 'OUTPUT_DIR' \
  -e 'VIS_DIR' \
  -e 'CONTROLNET_MODEL_PATH' \
  -e 'IP_ADAPTER_PATH' \
  -e 'ip-adapter_sd15.bin' \
  -e 'ip-adapter-plus_sd15.bin' \
  -e 'ip_adapter_with_style.bin' \
  -e 'num_inference_steps' \
  -e 'scale=' \
  -e 'image_generation.py' \
  -e 'image_generation_style.py' \
  image_generation*.sh logs/image_gen/*.err logs/image_gen/*.out \
  > docs/inventory/experiment_trace/image_gen_params_grep.txt 2>/dev/null

find image_generation -maxdepth 1 -mindepth 1 -printf '%TY-%Tm-%Td %TH:%TM:%TS  %p\n' | sort \
  > docs/inventory/experiment_trace/image_generation_dir_times.txt

find image_generation -maxdepth 2 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS  %p\n' | sort \
  > docs/inventory/experiment_trace/image_generation_file_times.txt

find logs/image_gen -maxdepth 1 -type f -printf '%TY-%Tm-%Td %TH:%TM:%TS  %p\n' | sort \
  > docs/inventory/experiment_trace/image_gen_log_times.txt
