"""Save model responses; do not send rubrics to the model or claim automatic accuracy."""
import argparse
import json
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--adapter')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-tokens',type=int,default=512)
    parser.add_argument('--cases',type=Path,default=ROOT/'evaluation/holdout.jsonl')
    args=parser.parse_args()
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    mx.random.seed(42)
    model,tokenizer=load(args.model,adapter_path=args.adapter)
    cases=[json.loads(x) for x in args.cases.read_text().splitlines()]
    system=(ROOT/'system_prompt.txt').read_text().strip()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as out:
        for case in cases:
            prompt=tokenizer.apply_chat_template([
                dict(role='system',content=system),dict(role='user',content=case['prompt'])],
                tokenize=True,add_generation_prompt=True,return_dict=False)
            start=time.perf_counter()
            pieces=[]; last=None
            for response in stream_generate(model,tokenizer,prompt=prompt,
                    max_tokens=args.max_tokens,sampler=make_sampler(temp=0)):
                pieces.append(response.text); last=response
            assert last is not None
            row=dict(id=case['id'],prompt=case['prompt'],text=''.join(pieces),
                adapted=bool(args.adapter),seed=42,temperature=0,max_tokens=args.max_tokens,
                prompt_tokens=last.prompt_tokens,generation_tokens=last.generation_tokens,
                finish_reason=last.finish_reason,peak_memory_gb=last.peak_memory,
                elapsed_seconds=time.perf_counter()-start)
            out.write(json.dumps(row,ensure_ascii=False)+'\n');out.flush()
            print(case['id'],row['finish_reason'],row['generation_tokens'],flush=True)

if __name__=='__main__': main()
