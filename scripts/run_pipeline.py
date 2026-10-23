"""Run the fixed small-scale experiment in a local macOS terminal with Metal access."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import yaml
from pathlib import Path
from download_model import download
from validate_data import validate
from check_numerics import check as check_numerics

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--data-dir',type=Path,default=ROOT/'data')
    parser.add_argument('--config',type=Path,default=ROOT/'train_config.yaml')
    parser.add_argument('--cases',type=Path,default=ROOT/'evaluation/holdout.jsonl')
    args=parser.parse_args()
    model=args.model.resolve(); run=args.run_dir.resolve()
    data_dir=args.data_dir.resolve();config=args.config.resolve();cases=args.cases.resolve()
    settings=yaml.safe_load(config.read_text())
    if any((run/name).exists() for name in ('baseline.log','before.jsonl','training.log','adapters')):
        parser.error('This run directory already contains experiment outputs. Use a new directory.')
    if args.offline:
        os.environ['HF_HUB_OFFLINE']='1'
        os.environ['TRANSFORMERS_OFFLINE']='1'
    run.mkdir(parents=True,exist_ok=True)
    status_path=run/'status.json'
    status={'status':'starting','started_at':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def write_status(): status_path.write_text(json.dumps(status,indent=2)+'\n')
    write_status()
    try:
        # Exit clearly in restricted/headless environments before downloading or training.
        import mlx.core as mx
        if not mx.metal.is_available(): raise RuntimeError('A local Metal GPU is required for this pipeline.')
        status['status']='downloading';write_status()
        download(model,offline=args.offline)
        status['status']='validating';write_status()
        report=validate(model,data_dir,cases,settings['max_seq_length'])
        report['numerical_checks']=check_numerics()
        (run/'data_checks.json').write_text(json.dumps(report,indent=2)+'\n')
        lock=json.loads((ROOT/'model_lock.json').read_text())
        metadata=dict(model=lock,platform=platform.platform(),python=platform.python_version(),
            versions={p:importlib.metadata.version(p) for p in ['mlx','mlx-lm','transformers','numpy']},
            hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [data_dir/'train.jsonl',data_dir/'valid.jsonl',cases,config,ROOT/'model_lock.json',ROOT/'scripts/train_mlx.py']},
            selection='Fixed final iteration; no checkpoint selection using diagnostic responses.')
        (run/'run_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        def execute(label,command):
            status['status']=label;write_status()
            print(label,flush=True);start=time.perf_counter()
            with (run/f'{label}.log').open('x') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,bufsize=1,env=dict(os.environ,PYTHONUNBUFFERED='1'))
                for line in process.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
            if code: raise RuntimeError(f'{label} failed with exit code {code}; inspect its log.')
            status[label+'_seconds']=time.perf_counter()-start;write_status()
        python=sys.executable
        execute('baseline',[python,str(ROOT/'scripts/evaluate_mlx.py'),'--model',str(model),'--cases',str(cases),'--output',str(run/'before.jsonl')])
        execute('training',[python,str(ROOT/'scripts/train_mlx.py'),'--config',str(config),
            '--model',str(model),'--data',str(data_dir),'--adapter-path',str(run/'adapters')])
        execute('adapted',[python,str(ROOT/'scripts/evaluate_mlx.py'),'--model',str(model),
            '--adapter',str(run/'adapters'),'--cases',str(cases),'--output',str(run/'after.jsonl')])
        status['status']='completed';status['completed_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();write_status()
        print('Completed. Adapter and before/after responses are ready for review.',flush=True)
    except Exception as exc:
        status['status']='failed';status['error']=f'{type(exc).__name__}: {exc}';write_status()
        raise

if __name__=='__main__': main()
