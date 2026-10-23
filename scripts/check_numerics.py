"""Check selected numerical answers using the actual code embedded in the data.

The examples in this repository are locally reviewed inputs, not arbitrary
model output. This script does not execute generated diagnostic responses.
"""
import ast
import json
import math
from pathlib import Path
import numpy as np
from train_mlx import target_mask

ROOT = Path(__file__).resolve().parents[1]


def example_functions():
    rows = [json.loads(line) for line in (ROOT / 'data/train.jsonl').read_text().splitlines()]
    functions = {}
    for row in rows:
        answer = row['messages'][-1]['content']
        if 'import numpy as np' not in answer:
            continue
        lines = answer[answer.index('import numpy as np'):].splitlines()
        # Prose follows the function without a code fence in these original examples.
        for end in range(len(lines), 0, -1):
            try:
                tree = ast.parse('\n'.join(lines[:end]))
            except SyntaxError:
                continue
            if not any(isinstance(node, ast.FunctionDef) for node in tree.body):
                continue
            if not all(isinstance(node, (ast.Import, ast.FunctionDef)) for node in tree.body):
                continue
            scope = {}
            exec(compile(tree, '<reviewed-training-example>', 'exec'), scope)
            functions.update({node.name: scope[node.name] for node in tree.body
                              if isinstance(node, ast.FunctionDef)})
            break
    return functions


def check():
    f = example_functions()
    required = {'orbital_unitary', 'density_matrix', 'orthogonality_error',
                'ci_residual', 'rhf_energy', 'central_directional_derivative'}
    assert required <= f.keys(), required - f.keys()
    theta = math.pi / 3
    kappa = np.array([[0, theta], [-theta, 0]])
    exact = np.array([[math.cos(theta), math.sin(theta)],
                      [-math.sin(theta), math.cos(theta)]])
    assert np.allclose(f['orbital_unitary'](kappa), exact, atol=1e-13)

    rng = np.random.default_rng(42)
    raw = rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5))
    unitary = f['orbital_unitary'](0.1 * (raw - raw.conj().T))
    overlap = np.diag([1., 2., 3., 4., 5.])
    coeff = np.diag(1 / np.sqrt(np.diag(overlap))) @ unitary
    assert f['orthogonality_error'](coeff, overlap) < 1e-12
    density = f['density_matrix'](coeff, [2, 2, 0, 0, 0])
    assert np.allclose(density, density.conj().T)
    assert abs(np.trace(density @ overlap) - 4) < 1e-12
    assert np.linalg.norm(density @ overlap @ density - 2 * density) < 1e-12

    # Known eigenpair of a Hermitian matrix with a nonzero off-diagonal entry.
    hamiltonian = np.array([[1., 1.], [1., 1.]])
    vector = np.array([1., 1.]) / np.sqrt(2)
    assert f['ci_residual'](lambda x: hamiltonian @ x, vector, 2.) < 1e-13
    assert f['ci_residual'](lambda x: hamiltonian @ x, vector, 1.) > 0.9

    # One doubly occupied spatial orbital: E = 2h + (11|11) + E_nuc.
    energy = f['rhf_energy'](np.array([[2.]]), np.array([[-1.]]),
                           np.array([[1.4]]), np.array([[1.4]]), 0.2)
    assert abs(energy - (-1.1)) < 1e-12

    def weighted_energy(x):
        weight = 1 / (1 + np.exp(-x))
        return weight * x*x + (1-weight) * (x-2)**2
    derivative = f['central_directional_derivative'](weighted_energy, 0., 1., 1e-5)
    assert abs(derivative + 3.) < 1e-8
    try:
        f['orbital_unitary'](np.eye(2))
    except ValueError:
        pass
    else:
        raise AssertionError('Non-anti-Hermitian input was accepted')
    mask=target_mask(np.array([[2,5],[4,6]]),7,np)
    assert np.array_equal(mask,np.array([[False,True,True,True,False,False,False],
                                        [False,False,False,True,True,False,False]]))
    return {'status': 'passed', 'checked_functions': sorted(required),
            'loss_mask_check': 'Passed: prompt and padding targets are excluded.',
            'scope': 'Selected NumPy examples and loss boundaries; no automatic scientific grading or PySCF execution.'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
