# Training procedure

## 1. Select the material

Use standard electronic-structure theory, published orbital-optimization work, public PySCF interfaces, and original numerical exercises. Record the source and the applicable software version. Preserve distinctions between a reference result, a derivation, and an implementation assumption.

## 2. Turn knowledge into tasks

Create question-answer pairs that require electron counting, explicit density conventions, derivations, code explanation, error diagnosis, or restrained scientific writing. An answer should state its assumptions and should not invent molecular energies, timings or references. Numerical examples are synthetic exercises, not measured molecular results.

The current build contains 88 training examples and 12 validation examples. Eight diagnostic prompts are kept outside the training files. Exact prompts are distinct, but topics and sources overlap. Parameterized exercises test limited transfer rather than broad generalization.

## 3. Check the data

Validate JSONL roles, nonempty messages, provenance references and exact duplicate prompts. Tokenize with the pinned model's actual chat template. The observed maximum training and validation lengths are 290 and 180 tokens, respectively, below the 1024-token limit. Check representative equations and code against numerical invariants such as orbital orthogonality and finite-difference derivatives.

These checks do not constitute independent expert review of every answer.

## 4. Record the baseline

Load the pinned MLX model without adapters. Use the same system prompt, tokenizer and diagnostic prompts that will be used after training. Decode greedily with seed 42 and a 512-token output limit. Save the complete response, token count, stopping reason, time and peak memory. Do not provide grading rubrics to the model.

## 5. Train

Freeze the quantized base parameters. Insert rank-8 LoRA matrices into the query and value projections of the last four transformer blocks. MLX evaluates a frozen linear branch plus a trainable low-rank branch:

`y = Q(W0)x + s B A x`, with `s = 16`.

Optimize the answer-token cross-entropy using Adam at a learning rate of `1e-4`. System, user and padding tokens are masked from the loss. The training entry point supplies an explicit mask with an exclusive sequence-length boundary to the MLX-LM trainer, including validation. Use batch size one, gradient checkpointing and 176 optimizer steps. Evaluate the full validation set every 44 steps. Save intermediate adapters for reproducibility, but use the fixed final adapter in the diagnostic comparison.

This is quantized-base LoRA in MLX. It does not reproduce all implementation details of the original NF4-based QLoRA recipe. The 176 steps represent two nominal passes over 88 records at batch size one, subject to the loader's batching behavior.

## 6. Evaluate and report

Load the same base checkpoint with the final adapter and repeat the fixed diagnostic protocol. Review the output against each rubric, check code API validity, and report partial answers and truncations separately. Report training loss, validation loss, actual runtime and memory from logs. Do not replace missing measurements with estimates.

Diagnostic responses are not used to select checkpoints or tune hyperparameters. The diagnostic set was visible during dataset development; report that limitation explicitly. An improvement on eight prompts would not establish broad domain expertise.

## 7. Package

Include original data, public references, exact configuration, dependency versions, reviewed responses, adapter weights, a workflow figure and the manuscript. Exclude base-model downloads, caches, machine-specific paths and unrelated local files. Scan the complete candidate file set before publishing. Keep a fresh version-control history for the release.
