#!/usr/bin/env bash
# 画像用エンコーダーを取得。Bonsai 派生版は親モデルのファイルを共有する。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

download() {
    local repo="$1" target="$2" file="$3"
    mkdir -p "$target/models"
    if [[ -s "$target/models/$file" ]]; then
        echo "取得済み: $target/models/$file"
        return
    fi
    curl --fail --location --retry 3 --continue-at - \
        "https://huggingface.co/$repo/resolve/main/$file" \
        --output "$target/models/$file.part"
    mv "$target/models/$file.part" "$target/models/$file"
}

download unsloth/Qwen3.5-9B-GGUF qwen3.5-9b mmproj-F16.gguf
download unsloth/Qwen3.6-35B-A3B-GGUF qwen3.6-35b-a3b mmproj-F16.gguf
download unsloth/Qwen3.8-27B-GGUF qwen3.8-27b mmproj-F16.gguf
download unsloth/gemma-4-26B-A4B-it-GGUF gemma4-26b-a4b mmproj-F16.gguf
download prism-ml/Ternary-Bonsai-2-27B-gguf ternary-bonsai-2-27b Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf
