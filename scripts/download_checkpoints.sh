#!/usr/bin/env bash
# scripts/download_checkpoints.sh — Download released checkpoints from GitHub Release.
# Run from repo root: bash scripts/download_checkpoints.sh
#
# Checkpoints are NOT stored in the git repo (see .gitignore *.pt pattern).
# They are attached to GitHub Release: v0.1-checkpoints
# https://github.com/Mahmoudreza/Rev_Max_discount_DeepL/releases/tag/v0.1-checkpoints
# NOTE: Release is pending — the script will print instructions if not yet published.
set -euo pipefail
cd "$(dirname "$0")/.."

# Release tag: v0.1-checkpoints
# GitHub repo: https://github.com/Mahmoudreza/Rev_Max_discount_DeepL
RELEASE_BASE_URL="https://github.com/Mahmoudreza/Rev_Max_discount_DeepL/releases/download/v0.1-checkpoints"
DEST="results/checkpoints"
mkdir -p "$DEST"

# Checkpoint catalog:
#   name                         sha256 (first 16 chars)   size   key experiment
declare -A CHECKSUMS=(
  ["rev_gnn_lstm.pt"]="8fbc4648ea4eda4e"           # 528K  Idea 1 main model
  ["rev_gnn_lstm_budget.pt"]="4b966e17b435fcd6"    # 528K  Idea 3 budget-aware
  ["rev_gnn_lstm_tc.pt"]="20901c29a714628c"        # 528K  Idea 2 time-critical
  ["rev_gnn_im_rl.pt"]="a8232ce2998e9aed"          # 200K  Ablation: IM+RL baseline
  ["rev_gail_lstm.pt"]="f77393ab7e1ac097"          # 528K  GAIL+LSTM ablation
  ["rev_gail_rl_rich.pt"]="8b64e55b62a0bd0a"       # 200K  GAIL+RL rich ablation
)

# Probe that the release actually exists before downloading anything
HTTP_CODE=$(curl -o /dev/null -s -w "%{http_code}" \
    "https://github.com/Mahmoudreza/Rev_Max_discount_DeepL/releases/tag/v0.1-checkpoints")
if [ "$HTTP_CODE" != "200" ]; then
    echo "Release v0.1-checkpoints not yet published (HTTP $HTTP_CODE)."
    echo ""
    echo "The .pt files will be attached to the GitHub Release once training is"
    echo "complete. In the meantime, train from scratch:"
    echo "  bash scripts/reproduce_all.sh   (SKIP_TRAINING=0)"
    echo ""
    echo "Or copy your local checkpoints:"
    echo "  cp /path/to/checkpoints/*.pt results/checkpoints/"
    echo ""
    echo "SHA256 checksums for manual verification:"
    for name in "${!CHECKSUMS[@]}"; do
        echo "  ${CHECKSUMS[$name]}  $name"
    done
    exit 1
fi

for name in "${!CHECKSUMS[@]}"; do
    dest_file="$DEST/$name"
    url="$RELEASE_BASE_URL/$name"
    expected_prefix="${CHECKSUMS[$name]}"

    if [ -f "$dest_file" ]; then
        actual=$(shasum -a 256 "$dest_file" | awk '{print $1}' | cut -c1-16)
        if [ "$actual" = "$expected_prefix" ]; then
            echo "  [ok] $name (already present, checksum match)"
            continue
        else
            echo "  [warn] $name checksum mismatch — re-downloading..."
        fi
    fi

    echo "  Downloading $name ..."
    curl -fsSL "$url" -o "$dest_file"
    actual=$(shasum -a 256 "$dest_file" | awk '{print $1}' | cut -c1-16)
    if [ "$actual" = "$expected_prefix" ]; then
        echo "  [ok] $name checksum verified"
    else
        echo "  [FAIL] $name: expected sha256 prefix $expected_prefix, got $actual"
        exit 1
    fi
done

echo ""
echo "All checkpoints downloaded and verified → $DEST/"
