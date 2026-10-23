"""Download a fixed public model revision without executing repository code."""
import argparse
import json
import os
import hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def verify_local(destination):
    destination=Path(destination).resolve()
    lock=json.loads((ROOT/'model_lock.json').read_text())
    manifest_path=destination/'verified_manifest.json'
    if not manifest_path.is_file(): return False
    manifest=json.loads(manifest_path.read_text())
    if any(manifest.get(k)!=lock[k] for k in ('repository','revision')):
        raise ValueError('Local model manifest does not match the pinned checkpoint.')
    required={'config.json','tokenizer.json','tokenizer_config.json','model.safetensors'}
    if not required.issubset({r['name'] for r in manifest['files']}):
        raise ValueError('Model manifest is incomplete.')
    for record in manifest['files']:
        name=record['name']
        if Path(name).name!=name: raise ValueError('Unexpected path in model manifest.')
        path=destination/name
        if not path.is_file() or path.stat().st_size!=record['size']: return False
        digest=hashlib.sha256() if record.get('sha256') else hashlib.sha1()
        if not record.get('sha256'):
            digest.update(b'blob '+str(record['size']).encode()+b'\0')
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(4*1024*1024),b''): digest.update(chunk)
        expected=record.get('sha256') or record.get('git_blob_sha1')
        if digest.hexdigest()!=expected: raise ValueError('Model checksum mismatch: '+name)
    return True

def download(destination,offline=False):
    destination=Path(destination).resolve()
    if verify_local(destination):
        print('Verified local checkpoint; no network request is required.',flush=True)
        return destination
    if offline:
        raise RuntimeError('A complete verified local checkpoint is required in offline mode.')
    os.environ.setdefault('HF_HOME',str(destination.parent/'hf-cache'))
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    from huggingface_hub import HfApi, snapshot_download
    lock=json.loads((ROOT/'model_lock.json').read_text())
    snapshot_download(lock['repository'],revision=lock['revision'],local_dir=destination,
        allow_patterns=['*.json','*.safetensors','*.txt'],max_workers=2)
    info=HfApi().model_info(lock['repository'],revision=lock['revision'],files_metadata=True)
    if info.sha!=lock['revision']:
        raise ValueError('Remote metadata does not match the pinned revision.')
    manifest=dict(repository=lock['repository'],revision=lock['revision'],files=[])
    for file in info.siblings:
        if not file.rfilename.endswith(('.json','.safetensors','.txt')): continue
        manifest['files'].append(dict(name=file.rfilename,size=file.size,
            sha256=file.lfs.sha256 if file.lfs else None,git_blob_sha1=file.blob_id))
    manifest_path=destination/'verified_manifest.json'
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    if not verify_local(destination):
        raise RuntimeError('Model download is incomplete; do not start training.')
    print('Checkpoint checksums verified; future runs can use --offline.',flush=True)
    return destination

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--offline',action='store_true')
    args=parser.parse_args()
    download(args.output,offline=args.offline)
