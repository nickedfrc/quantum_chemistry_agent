# Local Ollama export

The completed pilot was imported under `quantum-chemistry-pilot:1.5b` using Ollama 0.40.6. Loading and inference succeeded; the separate smoke answer was scientifically incorrect. Keep this experimental status visible when using it.

## Reproduce the conversion

Use a separate Python environment so export dependencies do not change the training environment. Download the official llama.cpp source and CPU quantizer for release `b11836` from [the upstream release](https://github.com/ggml-org/llama.cpp/releases/tag/b11836). Keep its `conversion` and `gguf-py` directories alongside `convert_hf_to_gguf.py`.

```bash
python3 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r requirements.export.lock.txt
python -m pip install --no-deps /path/to/llama.cpp/gguf-py
python scripts/download_model.py --output models/base
python scripts/merge_affine_lora.py --model models/base --adapter adapters --output models/fused
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python /path/to/llama.cpp/convert_hf_to_gguf.py models/fused --outtype f16 --outfile models/qchem-pilot-f16.gguf --model-name Quantum-Chemistry-Pilot-1.5B --use-temp-file
/path/to/llama-quantize models/qchem-pilot-f16.gguf models/qchem-pilot-q4_k_m.gguf Q4_K_M 4
ollama create quantum-chemistry-pilot:1.5b -f Modelfile
ollama run quantum-chemistry-pilot:1.5b
```

The placeholder paths refer to your upstream converter checkout and quantizer binary. The model download script additionally needs `huggingface_hub`, included by the export requirements. The original base-model license is retained in `licenses/`.

The CPU merger verifies the pinned base files, unpacks affine four-bit values in low-nibble-first order, reconstructs each group as `q * scale + bias`, and adds the MLX low-rank update. It uses the MLX direct scale of 16, not an alpha divided by rank. Blocks are written as FP16 Safetensors, with dimensions and finiteness checked. The official converter handles Qwen2 tensor names and the tokenizer. The final quantizer produces Q4_K_M GGUF.

Synthetic packing and low-rank algebra tests pass. The exported Safetensors file was also parsed by the independent Safetensors library. This CPU path was not compared bitwise against native MLX fusion. FP16 rounding and the second quantization can alter the outputs, so inference is checked separately. No deployment quality improvement is claimed.

The published export record contains source revision, adapter hash, merged hash, GGUF hash, module list, and tool versions. The original MLX adapter remains the reproducible trained artifact.

## Follow-up export

The same conversion was completed for the second adapter and imported as `quantum-chemistry-followup:1.5b`. For reproduction, pass `--adapter experiments/followup/adapters --output models/fused-followup` to the merger, convert that directory to `models/qchem-followup-f16.gguf`, quantize to `models/qchem-followup-q4_k_m.gguf`, and run `ollama create quantum-chemistry-followup:1.5b -f experiments/followup/Modelfile`. Use a new model tag to preserve the first pilot.

Its local inference call completed but produced an incorrect spin expression and repetition until the token limit. The export hashes and raw response are retained in `results/deployment-followup/`. This again verifies deployability, not scientific reliability.
