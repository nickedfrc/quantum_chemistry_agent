"""Dataset format, provenance and length checks; no training required."""
import argparse
import ast
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def validate(model=None, data_dir=None, cases_file=None, max_length=1024):
    data_dir=Path(data_dir) if data_dir else ROOT/'data'
    cases_file=Path(cases_file) if cases_file else ROOT/'evaluation/holdout.jsonl'
    sources={x['id'] for x in json.loads((ROOT/'sources.json').read_text())}
    metadata=json.loads((data_dir/'provenance.json').read_text())
    seen=set(); counts={}; lengths={}
    tokenizer=None
    if model:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(model),local_files_only=True,trust_remote_code=False)
    for split in ['train','valid']:
        rows=[json.loads(x) for x in (data_dir/f'{split}.jsonl').read_text().splitlines()]
        provenance=[x for x in metadata if x['split']==split]
        assert len(rows)==len(provenance)
        counts[split]=len(rows); lengths[split]=[]
        for row,meta in zip(rows,provenance):
            assert [m['role'] for m in row['messages']]==['system','user','assistant']
            assert all(isinstance(m['content'],str) and m['content'].strip() for m in row['messages'])
            prompt=row['messages'][1]['content']
            assert prompt not in seen
            seen.add(prompt)
            assert all(s in sources for s in meta['source_refs'])
            if tokenizer:
                tokens=tokenizer.apply_chat_template(row['messages'],tokenize=True,return_dict=False)
                assert isinstance(tokens,list) and all(isinstance(t,int) for t in tokens)
                length=len(tokens)
                prefix=tokenizer.apply_chat_template(row['messages'][:-1],tokenize=True,
                    add_generation_prompt=True,return_dict=False)
                assert tokens[:len(prefix)]==prefix, (meta['id'],'chat prefix mismatch')
                assert 0<len(prefix)<length, (meta['id'],'no supervised answer tokens')
                assert length<=max_length, (meta['id'],length)
                lengths[split].append(length)
    cases=[json.loads(x) for x in cases_file.read_text().splitlines()]
    assert all(x['prompt'] not in seen for x in cases)
    assert len({x['id'] for x in cases})==len(cases)
    for p in (ROOT/'scripts').glob('*.py'): ast.parse(p.read_text())
    report=dict(counts=counts,diagnostic_cases=len(cases),max_tokens={k:max(v) if v else None for k,v in lengths.items()},
        note='Exact prompt separation only; topics and sources overlap. Not an independent generalization benchmark.')
    print(json.dumps(report,indent=2))
    return report

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--model',type=Path)
    parser.add_argument('--data-dir',type=Path);parser.add_argument('--cases',type=Path)
    parser.add_argument('--max-length',type=int,default=1024)
    args=parser.parse_args();validate(args.model,args.data_dir,args.cases,args.max_length)
