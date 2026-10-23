"""Original, small public-method examples, executed with PySCF 2.6.2.

Run: python examples/verified_cases.py --output results/code-cases.json
All energies are in Hartree. These are educational checks, not benchmarks.
"""
import argparse
import json
import platform
from pathlib import Path
import time

import numpy as np
import scipy
from scipy.linalg import eigh
import pyscf
from pyscf import gto, scf, mp, cc, mcscf, ao2mo, lib


def rhf_water():
    mol = gto.M(atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
                unit='Angstrom', basis='sto-3g', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-11).run()
    assert mf.converged
    d, s, h = mf.make_rdm1(), mf.get_ovlp(), mf.get_hcore()
    j, k = mf.get_jk(mol, d)
    f = h + j - 0.5 * k
    reconstructed = 0.5 * np.einsum('ij,ji', d, h + f) + mol.energy_nuc()
    n = float(np.trace(d @ s))
    residual = float(np.linalg.norm(f @ d @ s - s @ d @ f))
    assert abs(n - mol.nelectron) < 1e-9
    assert abs(reconstructed - mf.e_tot) < 1e-9
    assert residual < 1e-5
    return dict(energy_hartree=float(mf.e_tot), electrons=n,
                reconstruction_error=float(abs(reconstructed-mf.e_tot)), scf_residual=residual)


def mp2_hf():
    mol = gto.M(atom='H 0 0 0; F 0 0 0.92', unit='Angstrom',
                basis='sto-3g', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-11).run()
    assert mf.converged
    calc = mp.MP2(mf)
    e_corr, t2 = calc.kernel()
    dm_mo = calc.make_rdm1()
    assert abs(calc.e_tot - (mf.e_tot + e_corr)) < 1e-10
    assert abs(np.trace(dm_mo) - mol.nelectron) < 1e-8
    return dict(hf_energy=float(mf.e_tot), correlation_energy=float(e_corr),
                total_energy=float(calc.e_tot), mo_density_trace=float(np.trace(dm_mo)),
                amplitude_shape=list(t2.shape))


def ccsd_h2():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', unit='Angstrom',
                basis='cc-pvdz', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-11).run()
    assert mf.converged
    calc = cc.CCSD(mf).set(conv_tol=1e-10)
    e_corr, t1, t2 = calc.kernel()
    assert calc.converged
    assert abs(calc.e_tot - mf.e_tot - e_corr) < 1e-9
    return dict(hf_energy=float(mf.e_tot), correlation_energy=float(e_corr),
                total_energy=float(calc.e_tot), singles_norm=float(np.linalg.norm(t1)),
                doubles_norm=float(np.linalg.norm(t2)))


def casci_lih():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', unit='Angstrom',
                basis='sto-3g', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-11).run()
    assert mf.converged
    calc = mcscf.CASCI(mf, 2, (1, 1))
    calc.fcisolver.conv_tol = 1e-11
    calc.kernel()
    assert calc.converged
    dm_active = calc.fcisolver.make_rdm1(calc.ci, calc.ncas, calc.nelecas)
    spin_squared, multiplicity = calc.fcisolver.spin_square(calc.ci, calc.ncas, calc.nelecas)
    assert abs(np.trace(dm_active) - 2) < 1e-9
    assert abs(spin_squared) < 1e-7
    return dict(total_energy=float(calc.e_tot), core_orbitals=int(calc.ncore),
                active_trace=float(np.trace(dm_active)), spin_squared=float(spin_squared),
                multiplicity=float(multiplicity), natural_occupations=np.linalg.eigvalsh(dm_active).tolist())


def casscf_lih():
    mol = gto.M(atom='Li 0 0 0; H 0 0 1.6', unit='Angstrom',
                basis='sto-3g', charge=0, spin=0, verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-11).run()
    assert mf.converged
    fixed = mcscf.CASCI(mf, 2, (1, 1)).run()
    optimized = mcscf.CASSCF(mf, 2, (1, 1))
    optimized.conv_tol = 1e-9
    optimized.conv_tol_grad = 1e-5
    optimized.kernel()
    assert fixed.converged and optimized.converged
    assert optimized.e_tot <= fixed.e_tot + 1e-7
    return dict(casci_energy=float(fixed.e_tot), casscf_energy=float(optimized.e_tot),
                orbital_relaxation=float(optimized.e_tot-fixed.e_tot), core_orbitals=int(optimized.ncore))


def uhf_hydrogen():
    mol = gto.M(atom='H 0 0 0', basis='cc-pvdz', charge=0, spin=1, verbose=0)
    mf = scf.UHF(mol).set(conv_tol=1e-11).run()
    assert mf.converged
    da, db = mf.make_rdm1()
    s = mf.get_ovlp()
    na, nb = float(np.trace(da @ s)), float(np.trace(db @ s))
    spin_squared, multiplicity = mf.spin_square()
    assert abs(na-1) < 1e-9 and abs(nb) < 1e-9
    assert abs(spin_squared-0.75) < 1e-9
    return dict(total_energy=float(mf.e_tot), alpha_electrons=na, beta_electrons=nb,
                spin_squared=float(spin_squared), multiplicity=float(multiplicity))


def integral_transform():
    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', unit='Angstrom', basis='sto-3g', verbose=0)
    mf = scf.RHF(mol).set(conv_tol=1e-11).run()
    assert mf.converged
    c = mf.mo_coeff
    ao = mol.intor('int2e', aosym='s1')
    explicit = np.einsum('up,vq,wr,xs,uvwx->pqrs', c, c, c, c, ao, optimize=True)
    packed = ao2mo.kernel(mol, c)
    restored = ao2mo.restore(1, packed, c.shape[1])
    error = float(np.max(np.abs(explicit-restored)))
    assert error < 1e-10
    return dict(full_shape=list(restored.shape), packed_shape=list(packed.shape), max_error=error)


def generalized_eigenproblem():
    s = np.array([[1.0, 0.2], [0.2, 1.4]])
    f = np.array([[-1.1, 0.1], [0.1, 0.4]])
    energies, c = eigh(f, s)
    residual = float(np.linalg.norm(f @ c - (s @ c) * energies))
    metric_error = float(np.linalg.norm(c.T @ s @ c - np.eye(2)))
    assert residual < 1e-12 and metric_error < 1e-12
    return dict(eigenvalues=energies.tolist(), residual=residual, metric_error=metric_error)


def diis_coefficients():
    errors = np.array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    gram = errors @ errors.T
    augmented = np.zeros((4, 4))
    augmented[:3, :3] = gram
    augmented[:3, 3] = augmented[3, :3] = -1
    rhs = np.array([0., 0., 0., -1.])
    coeff = np.linalg.solve(augmented, rhs)[:3]
    assert abs(coeff.sum()-1) < 1e-12
    assert np.linalg.norm(gram @ coeff - np.ones(3)*(gram @ coeff)[0]) < 1e-12
    return dict(coefficients=coeff.tolist(), sum=float(coeff.sum()),
                extrapolated_error_norm=float(np.linalg.norm(coeff @ errors)))


def rayleigh_ritz():
    h = np.diag([0., 1., 3., 6.]) + 0.05*np.ones((4, 4))
    trial = np.array([[1., 0.], [0.2, 1.], [0.1, 0.3], [0.4, 0.2]])
    q, _ = np.linalg.qr(trial)
    projected = q.T @ h @ q
    energies, vectors = np.linalg.eigh(projected)
    vector = q @ vectors[:, 0]
    residual = h @ vector - energies[0] * vector
    exact = np.linalg.eigvalsh(h)[0]
    assert energies[0] >= exact - 1e-12
    assert np.linalg.norm(q.T @ residual) < 1e-12
    return dict(ritz_energy=float(energies[0]), exact_lowest=float(exact),
                full_residual=float(np.linalg.norm(residual)), projected_residual=float(np.linalg.norm(q.T @ residual)))


def gradient_check():
    def solve(distance):
        mol = gto.M(atom=f'H 0 0 0; F 0 0 {distance}', unit='Bohr',
                    basis='sto-3g', charge=0, spin=0, verbose=0)
        mf = scf.RHF(mol).set(conv_tol=1e-12).run()
        assert mf.converged
        return mf
    distance, step = 1.8, 1e-3
    mf = solve(distance)
    analytic = float(mf.nuc_grad_method().kernel()[1, 2])
    numeric = (solve(distance+step).e_tot-solve(distance-step).e_tot)/(2*step)
    assert abs(analytic-numeric) < 2e-6
    return dict(analytic_hartree_per_bohr=analytic, finite_difference=float(numeric),
                absolute_error=float(abs(analytic-numeric)), step_bohr=step)


def density_fitting_check():
    mol = gto.M(atom='H 0 0 0; F 0 0 0.92', unit='Angstrom',
                basis='cc-pvdz', charge=0, spin=0, verbose=0)
    conventional = scf.RHF(mol).set(conv_tol=1e-11).run()
    fitted = scf.RHF(mol).density_fit(auxbasis='cc-pvdz-jkfit').set(conv_tol=1e-11).run()
    assert conventional.converged and fitted.converged
    return dict(conventional_energy=float(conventional.e_tot), fitted_energy=float(fitted.e_tot),
                difference_hartree=float(fitted.e_tot-conventional.e_tot), auxiliary_basis='cc-pvdz-jkfit')


CASES = [rhf_water, mp2_hf, ccsd_h2, casci_lih, casscf_lih, uhf_hydrogen,
         integral_transform, generalized_eigenproblem, diis_coefficients, rayleigh_ritz,
         gradient_check, density_fitting_check]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    lib.num_threads(1)
    report = dict(pyscf=pyscf.__version__, numpy=np.__version__, scipy=scipy.__version__,
                  python=platform.python_version(), thread_limit_requested=1,
                  timing_note='Single short verification runs, not performance measurements.', cases=[])
    for case in CASES:
        start = time.perf_counter()
        result = case()
        report['cases'].append(dict(id=case.__name__, result=result, checks='passed',
                                    wall_seconds=time.perf_counter()-start))
        print(case.__name__, json.dumps(result), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
