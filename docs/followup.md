# Expanded follow-up experiment

Status: training and matched evaluation completed on October 23, 2026. The original pilot remains preserved as a separate experiment.

The follow-up adds 56 training records and 8 validation records, bringing the totals to 144 and 20. The added material includes original methods paragraphs, algorithm explanations, executable examples, code repairs, and interpretation of actual numerical outputs. Nine reference entries were added: three papers and six official documentation/API sources. Four paper references were consulted for the expansion; the Pulay entry was already present. Some classic paper records were checked only at metadata or abstract level; the source inventory states this explicitly.

## Verified code

`examples/verified_cases.py` contains 12 small original cases. They were executed with PySCF 2.6.2, NumPy 2.5.4 and SciPy 1.18.1. All assertions passed. The cases cover RHF, UHF, MP2, CCSD, CASCI, CASSCF, integral transformations, generalized eigenproblems, DIIS, Rayleigh–Ritz projection, nuclear gradients and density fitting. Measured values are in `results/code-cases.json`; short-run wall times are not performance benchmarks.

The training data include nine of these code families. Rayleigh–Ritz examples are assigned to validation. Nuclear-gradient and density-fitting families are held out of the new training records and used in six follow-up diagnostic prompts. Broad concepts can overlap with the original pilot, and all questions were prepared by the same AI assistant. This is a more deliberate family separation, not a blind or independently reviewed benchmark.

```bash
python3 -m venv .venv-science
source .venv-science/bin/activate
python -m pip install -r requirements.science.lock.txt
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python examples/verified_cases.py --output results/code-cases.json
python scripts/build_followup.py
```

The installed PySCF wheel reported that OpenMP was unavailable. Thread-limit requests and the actual environment are recorded separately; no speedup claim is based on these runs.

## Training protocol

Start fresh from the same pinned base, using the same rank, direct scale, last-four-block adapter placement, optimizer, and answer-only loss as the pilot. The fixed 576 updates at batch size one correspond to four nominal passes through 144 training records. Training and validation examples fit the 1024-token sequence limit; their observed maxima are 479 and 402 tokens.

```bash
python scripts/validate_data.py --model models/base --data-dir experiments/followup --cases experiments/followup/holdout.jsonl
python scripts/run_pipeline.py --model models/base --run-dir runs/followup-01 --offline --data-dir experiments/followup --config experiments/followup/train_config.yaml --cases experiments/followup/holdout.jsonl
```

Run this in the training environment with Metal access. The pipeline evaluates the fresh base, trains, then evaluates the fixed final checkpoint using the six new prompts. It saves raw responses and hashes. Diagnostic rubrics are not sent to the model. The extra data and longer training schedule both change, so any future difference cannot be attributed to data size alone. The saved outputs have now been reviewed; no scientific-correctness improvement was established.

## References consulted for the expansion

- Sun et al., [Recent developments in the PySCF program package](https://arxiv.org/abs/2002.12531), J. Chem. Phys. 153, 024109 (2020).
- Sun, [Libcint](https://arxiv.org/abs/1412.0649), J. Comput. Chem. 36, 1664–1671 (2015).
- Pulay, [Improved SCF convergence acceleration](https://doi.org/10.1002/jcc.540030413), J. Comput. Chem. 3, 556–560 (1982).
- Davidson, [Iterative calculation of low eigenpairs](https://doi.org/10.1016/0021-9991(75)90065-0), J. Comput. Phys. 17, 87–94 (1975).
- Official PySCF guides for [SCF](https://pyscf.org/user/scf.html), [MP2](https://pyscf.org/user/mp.html), [CC](https://pyscf.org/user/cc.html), [integral transformation](https://pyscf.org/_modules/pyscf/ao2mo.html), [density fitting](https://pyscf.org/user/df.html), and [geometry optimization](https://pyscf.org/user/geomopt.html).

The textbook references from the original pilot remain in use. Full books and papers are not copied into the dataset or repository.

## Measured outcome

Training completed 576 steps in 449.09 seconds. Validation losses at steps 1, 144, 288, 432 and 576 were 3.388, 2.769, 2.578, 2.495 and 2.466. Strict diagnostic passes remained 0/6 before and after; truncations decreased from 3/6 to 1/6, and generated tokens from 2,766 to 1,312. Lower loss and shorter answers did not establish correct reasoning or valid code. The training-memory high-water mark was not recorded.

Two AI-assisted reviews agreed on the six case assessments; this is not human expert adjudication. Raw outputs, case reasons, input hashes and logs are in `results/followup-01/`. The final adapter is in `experiments/followup/adapters/`. No generated dependency list or generated scientific code was executed.

```bash
python -m mlx_lm generate --model models/base --adapter-path experiments/followup/adapters --prompt "Explain the AO overlap metric in RHF."
```
