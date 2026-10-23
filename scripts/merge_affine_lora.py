"""CPU-only export of this pilot's MLX affine-int4 base and LoRA to FP16.

This intentionally supports only the pinned Qwen2 pilot format. It streams
weights to limit memory use. The output is for deployment, not a replacement
for the matched MLX evaluation. GPU kernels can round differently.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct

import numpy as np


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class TensorFile:
    def __init__(self, path):
        self.path = Path(path)
        with self.path.open('rb') as f:
            size = struct.unpack('<Q', f.read(8))[0]
            if size > 10_000_000:
                raise ValueError('Unexpected tensor header size')
            self.header = json.loads(f.read(size))
        self.start = 8 + size

    def array(self, name):
        entry = self.header[name]
        dtype = {'F16': '<f2', 'F32': '<f4', 'U32': '<u4'}[entry['dtype']]
        start, end = entry['data_offsets']
        expected = math.prod(entry['shape']) * np.dtype(dtype).itemsize
        if end - start != expected or self.start + end > self.path.stat().st_size:
            raise ValueError('Invalid tensor offsets')
        return np.memmap(self.path, mode='r', dtype=dtype,
                         offset=self.start + start, shape=tuple(entry['shape']))


def dequantize(packed, scales, biases, group_size=64):
    """Low nibble first; affine reconstruction q * scale + bias."""
    q = ((packed[..., None] >> (4 * np.arange(8, dtype=np.uint32))) & 15)
    q = q.reshape(packed.shape[0], -1).astype(np.float32)
    q = q.reshape(packed.shape[0], -1, group_size)
    return (q * scales.astype(np.float32)[..., None] +
            biases.astype(np.float32)[..., None]).reshape(packed.shape[0], -1).astype('<f2')


def self_test():
    packed = np.array([[0x76543210, 0xFEDCBA98] * 4], dtype=np.uint32)
    got = dequantize(packed, np.array([[0.5]]), np.array([[-2.0]]))
    np.testing.assert_array_equal(got, (np.tile(np.arange(16), 4) * 0.5 - 2)[None])
    rng = np.random.default_rng(42)
    x, a, b = rng.normal(size=(3, 12)), rng.normal(size=(12, 8)), rng.normal(size=(8, 6))
    np.testing.assert_allclose(x @ ((16 * b.T) @ a.T).T, 16 * ((x @ a) @ b), rtol=1e-12, atol=1e-12)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    self_test()
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / 'evaluation/base_checkpoint_files.json').read_text())
    # Verify against the published pinned file inventory before merging.
    records = manifest['files']
    if isinstance(records, dict):
        records = [dict(v, name=k) for k, v in records.items()]
    for entry in records:
        name = entry.get('name', entry.get('path', entry.get('filename')))
        source = args.model / name
        if source.stat().st_size != entry['size']:
            raise ValueError('Base checkpoint size mismatch: ' + name)
        if entry.get('sha256'):
            valid = digest(source) == entry['sha256']
        else:
            content = source.read_bytes()
            valid = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == entry['git_blob_sha1']
        if not valid:
            raise ValueError('Base checkpoint hash mismatch: ' + name)
    config = json.loads((args.model / 'config.json').read_text())
    quant = config['quantization']
    if quant.get('bits') != 4 or quant.get('group_size') != 64 or quant.get('mode', 'affine') != 'affine':
        raise ValueError('Only affine int4, group size 64 is supported')
    ac = json.loads((args.adapter / 'adapter_config.json').read_text())
    scale = ac['lora_parameters']['scale']
    if scale != 16 or ac['lora_parameters']['rank'] != 8:
        raise ValueError('Unexpected pilot adapter configuration')
    base = TensorFile(args.model / 'model.safetensors')
    adapter = TensorFile(args.adapter / 'adapters.safetensors')
    modules = [f'model.layers.{layer}.self_attn.{proj}_proj' for layer in range(24, 28) for proj in ('q', 'v')]
    expected = {module + suffix for module in modules for suffix in ('.lora_a', '.lora_b')}
    if set(adapter.header) - {'__metadata__'} != expected:
        raise ValueError('Unexpected adapter tensors')
    output_header = {'__metadata__': {'format': 'pt'}}
    offset = 0
    names = sorted(k for k in base.header if k != '__metadata__' and not k.endswith(('.scales', '.biases')))
    for name in names:
        entry = base.header[name]
        shape = list(entry['shape'])
        if entry['dtype'] == 'U32':
            if not name.endswith('.weight') or len(shape) != 2:
                raise ValueError('Unsupported packed tensor')
            shape[-1] *= 8
        elif entry['dtype'] != 'F16':
            raise ValueError('Unsupported base dtype')
        length = math.prod(shape) * 2
        output_header[name] = dict(dtype='F16', shape=shape, data_offsets=[offset, offset + length])
        offset += length
    raw = json.dumps(output_header, separators=(',', ':')).encode()
    raw += b' ' * (-len(raw) % 8)
    args.output.mkdir(parents=True, exist_ok=False)
    target = args.output / 'model.safetensors'
    modified = []
    with target.open('xb') as out:
        out.write(struct.pack('<Q', len(raw)))
        out.write(raw)
        for name in names:
            weight = base.array(name)
            module = name.removesuffix('.weight')
            packed = weight.dtype == np.dtype('<u4')
            if packed:
                scales = base.array(module + '.scales')
                biases = base.array(module + '.biases')
                if scales.shape != biases.shape or scales.shape != (weight.shape[0], weight.shape[1] // 8):
                    raise ValueError('Inconsistent group layout')
            a = b = None
            if module in modules:
                a = adapter.array(module + '.lora_a')
                b = adapter.array(module + '.lora_b')
                shape = output_header[name]['shape']
                if a.shape != (shape[1], 8) or b.shape != (8, shape[0]):
                    raise ValueError('Adapter dimensions do not match')
                modified.append(module)
            for start in range(0, weight.shape[0], 512):
                stop = start + 512
                block = dequantize(weight[start:stop], scales[start:stop], biases[start:stop]) if packed else np.array(weight[start:stop], dtype='<f2')
                if a is not None:
                    delta = ((scale * b[:, start:stop].T) @ a.T).astype('<f2')
                    block = (block + delta).astype('<f2')
                if not np.isfinite(block).all():
                    raise ValueError('Non-finite output: ' + name)
                out.write(block.tobytes(order='C'))
            print(name, flush=True)
    if target.stat().st_size != 8 + len(raw) + offset or sorted(modified) != sorted(modules):
        raise ValueError('Incomplete export')
    config.pop('quantization')
    config.pop('quantization_config', None)
    config['torch_dtype'] = 'float16'
    (args.output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    for name in ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'generation_config.json'):
        if (args.model / name).exists():
            shutil.copyfile(args.model / name, args.output / name)
    report = dict(method='CPU NumPy affine int4 dequantization and FP16 LoRA merge',
                  source_revision=manifest.get('revision'), adapter_sha256=digest(args.adapter / 'adapters.safetensors'),
                  output_sha256=digest(target), output_bytes=target.stat().st_size,
                  tensor_count=len(names), parameter_count=offset // 2, merged_modules=modified,
                  synthetic_tests='passed', all_output_weights_finite=True,
                  limitation='Not bitwise compared with native MLX fusion; deployment is evaluated separately.')
    (args.output / 'merge_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
