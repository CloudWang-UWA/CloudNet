#!/usr/bin/env bash

set -euo pipefail

OUT_DIR="${1:-inventory}"
MAX_DEPTH="${MAX_DEPTH:-4}"
TOP_FILES="${TOP_FILES:-200}"

TARGETS=(
  "data"
  "cwang_model"
  "ip_adapter_models"
  "train_ip_adapter"
  "evaluation"
  "logs"
  "clip"
  "wandb"
)

mkdir -p "$OUT_DIR"

echo "Writing inventory to: $OUT_DIR"
echo "Max depth: $MAX_DEPTH"
echo "Top files per target: $TOP_FILES"

for target in "${TARGETS[@]}"; do
  if [[ ! -e "$target" ]]; then
    echo "Skipping missing target: $target"
    continue
  fi

  safe_name="${target//\//__}"

  echo
  echo "Scanning $target"

  find "$target" -maxdepth "$MAX_DEPTH" | sort \
    > "$OUT_DIR/${safe_name}_tree.txt"

  if [[ -d "$target" ]]; then
    du -sh "$target"/* 2>/dev/null | sort -h \
      > "$OUT_DIR/${safe_name}_top_sizes.txt" || true

    du -ah "$target" 2>/dev/null | sort -h | tail -n "$TOP_FILES" \
      > "$OUT_DIR/${safe_name}_largest_paths.txt" || true

    find "$target" -maxdepth 2 -type d | sort \
      > "$OUT_DIR/${safe_name}_dirs_depth2.txt"
  else
    ls -lh "$target" > "$OUT_DIR/${safe_name}_top_sizes.txt"
    ls -lh "$target" > "$OUT_DIR/${safe_name}_largest_paths.txt"
    printf '%s\n' "$target" > "$OUT_DIR/${safe_name}_dirs_depth2.txt"
  fi
done

{
  echo "Generated on: $(date)"
  echo "Host: $(hostname)"
  echo
  echo "Top-level sizes:"
  for target in "${TARGETS[@]}"; do
    if [[ -e "$target" ]]; then
      du -sh "$target" 2>/dev/null || true
    fi
  done | sort -h
} > "$OUT_DIR/_summary.txt"

echo
echo "Done. Files written under $OUT_DIR/"
