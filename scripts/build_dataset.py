"""Build original, small-scale exercises from standard electronic-structure topics."""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
records = json.loads((ROOT / 'data/seed_examples.json').read_text())
SYSTEM = (ROOT / 'system_prompt.txt').read_text().strip()

def add(identifier, question, answer, refs, split='train', topic='theory'):
    records.append(dict(id=identifier, split=split, topic=topic, source_refs=refs,
        review_status='assistant_authored; references_checked; not_independent_expert_review',
        messages=[dict(role='system', content=SYSTEM), dict(role='user', content=question),
                  dict(role='assistant', content=answer)]))

concepts = [
('What does a closed-shell RHF determinant assume?', 'Each occupied spatial orbital contains one alpha and one beta electron sharing the same spatial function. For an even-electron system, D=2 C_occ C_occ† is the spin-summed AO density. This ansatz can become inadequate during bond dissociation; SCF convergence alone does not establish physical accuracy.', 'book:10.7'),
('区分 RHF、ROHF 与 UHF。', 'RHF 通常用于闭壳层，共用成对电子的空间轨道。ROHF 用受限开壳层形式描述未成对电子；对于合适的自旋耦合可构造自旋纯参考。UHF 允许 alpha、beta 电子具有不同空间轨道，但一般不保证 S² 本征性。选择应依据电子数、自旋与相关性，不能只看哪个能量更低。', 'book:2.4'),
('Does a converged UHF calculation guarantee the intended total spin?', 'No. A fixed Nalpha-Nbeta fixes the spin projection. A UHF determinant may mix different total-spin sectors. Inspect <S²>, compare with S(S+1), and distinguish numerical convergence from spin purity.', 'book:2.4'),
('How are multiplicity and the total-spin quantum number related?', 'Multiplicity is 2S+1. Singlet, doublet, triplet and quartet correspond to S=0, 1/2, 1 and 3/2. Electron number and available orbitals constrain which values are possible. The spin projection M_S alone does not identify S.', 'book:2.4'),
('轨道能差是否就是总能量差？', '通常不是。轨道能包含平均场相互作用，而多电子总能量需要处理相互作用的重复计数。轨道松弛和电子相关也会影响激发或电离能。必须说明所采用的近似，不能把任意轨道能差直接称为精确激发能。', 'book:10.6'),
('Why does diagonalizing a single Fock matrix not complete a Hartree-Fock calculation?', 'The Fock matrix depends on the density built from its occupied eigenvectors. One diagonalization solves the eigenproblem for the current density. Self-consistency requires iterating density construction and Fock construction until the chosen energy, density and residual criteria are satisfied.', 'book:10.6'),
('非正交 AO 基中 Fock 本征方程是什么？', 'F C=S C epsilon，其中 S 是 AO 重叠矩阵，C†SC=I。直接对 F 使用普通正交本征求解会忽略 S。可以使用广义厄米本征求解器，或先构造正交化变换并处理近线性相关。', 'book:10.6'),
('How does symmetric orthogonalization work?', 'For positive-definite S=V diag(s) V†, define X=V diag(s^(-1/2)) V†. Then X† S X=I. Diagonalize X† F X and transform back. Small overlap eigenvalues require an explicit threshold or reduced basis; never divide blindly by almost-zero eigenvalues.', 'book:10.6'),
('Give the RHF Fock and energy formulas using a spin-summed density.', 'For the usual spatial-AO convention D=2 C_occ C_occ†, F=h+J[D]−K[D]/2. The electronic energy is E_elec=Tr[D(h+F)]/2, and the total energy adds nuclear repulsion. State the density and integral conventions before using these factors.', 'book:10.7'),
('Is a fractionally occupied one-particle density necessarily idempotent?', 'No. A mixed-state or correlated one-particle density can have fractional natural occupations and is generally not idempotent. The eigenvalue bounds, trace and relevant representability conditions still matter. Do not impose single-determinant idempotency on every ensemble or correlated density.', 'book:10.7'),
('C†SC=I 时，能把 C†C 也认为是 I 吗？', '不能。S 是 AO 度量，非正交 AO 基通常 S≠I。C†SC=I 表示分子轨道的物理正交归一，C†C 是系数向量的欧氏内积，两者不能混同。', 'book:3.1'),
('Why does orbital optimization use an anti-Hermitian generator?', 'If κ†=−κ, exp(κ) is unitary, so right multiplication preserves orthonormal molecular orbitals in a fixed AO metric. Independent variables can be chosen in one triangular part. Real orbitals use a real antisymmetric generator.', 'book:3.1'),
('Can every real orthogonal matrix be written as the exponential of a real antisymmetric matrix?', 'Not every orthogonal matrix. The exponential has determinant exp(Tr κ)=1 for real antisymmetric κ, so it lies in the special orthogonal group. An orthogonal matrix with determinant −1 also needs a reflection or a sign change. Continuous orbital rotations use the determinant-one component.', 'book:3.1'),
('Can exp(A+B) always be replaced by exp(A) exp(B)?', 'No. Commutation, [A,B]=0, is a sufficient condition for that identity. Orbital-rotation generators from different steps generally do not commute. Preserve the ordered product or use the appropriate Baker-Campbell-Hausdorff analysis.', 'book:3.1'),
('Taylor 截断后的轨道指数有什么数值风险？', '截断矩阵指数通常不再严格酉。误差随生成元大小与截断阶数变化，可能破坏轨道正交性。监测 ||U†U−I|| 与 ||C†SC−I||，使用稳定的矩阵指数算法或明确控制误差；不要把截断近似称为精确更新。', 'book:3.1'),
('What is the inverse of exp(κ)?', 'It is exp(−κ). If κ is anti-Hermitian, this is also exp(κ)†. The identity follows because κ commutes with −κ; it does not justify splitting arbitrary noncommuting sums inside an exponential.', 'book:3.1'),
('什么是直接 CI 求解？', '通过对试探 CI 向量计算 Hc 来进行迭代本征求解，避免显式存储完整哈密顿矩阵。Davidson 等方法用小子空间近似目标本征对。需要控制本征残差、子空间尺寸与正交性；直接法并不消除 CI 空间本身的组合增长。', 'book:11'),
('Are CASCI and CASSCF different names for the same calculation?', 'No. CASCI optimizes CI coefficients in a fixed orbital basis. CASSCF also optimizes orbitals for the selected objective. A CASCI calculation in previously optimized orbitals remains a fixed-orbital CI calculation at that stage.', 'book:12.1'),
('State the variational principle for the lowest eigenvalue in a finite CI space.', 'For a Hermitian H and nonzero c, the Rayleigh quotient c†Hc/(c†c) is at least the lowest eigenvalue in that same space. This does not make a truncated-space result exact in the full electronic Hilbert space. Higher roots need orthogonality or equivalent eigenproblem constraints.', 'book:11'),
('Why should the CI residual be reported?', 'An energy alone does not establish that an eigenvector is converged. For normalized c, evaluate r=Hc−Ec and compare its norm with a stated tolerance. Root identity and orthogonality also matter when several states are close.', 'book:11'),
('自然轨道占据接近1一定意味着某个确定自旋态吗？', '不能。自然占据反映一粒子密度的结构，单靠占据数通常不能确定多电子自旋耦合。还需要波函数、自旋适配信息或 S²。不要把单占据轨道个数与唯一多重度直接等同。', 'book:2.4'),
('Can CASSCF remove all dynamic correlation?', 'Generally no. CASSCF treats correlation within the selected active space and optimizes orbitals, but excitations outside that space remain restricted. Depending on the problem, an additional correlation treatment may be necessary. The active space and target observable must be stated.', 'book:12.1'),
('态平均轨道为什么可用于多个态？', '多个态共享同一轨道组，并以预先定义的加权能量作为优化目标。它为这些态提供共同表示，但每一态通常不各自处于单态最优轨道。报告时应区分平均能量、各态能量和权重。', 'book:12.7.2'),
('What must be specified before reproducing a state-averaged calculation?', 'Specify geometry and units, charge and spin sectors, basis, active electrons and orbitals, targeted roots, weights, initial orbitals, symmetry restrictions, convergence criteria, software version and root-tracking procedure. A method name alone is insufficient.', 'pyscf:mcscf_user'),
('两个态在迭代中交换能量排序，如何判断是否跟踪到了同一态？', '不能只使用根序号。比较相邻步骤的波函数或态重叠、对称性、主组态和物理性质，必要时采用态跟踪。接近简并时应关注相关子空间；相位和旋转不确定性也须处理。', 'book:12.7.2'),
('Does DIIS guarantee a decreasing total energy at every step?', 'No. DIIS minimizes an extrapolated residual norm in a subspace, not necessarily the actual nonlinear energy along the extrapolation. Use suitable convergence checks and safeguards; do not infer monotonicity from the name of the algorithm.', 'book:10.6.2'),
('为什么 DIIS 矩阵可能病态？', '若历史误差向量近线性相关，其 Gram 矩阵可能病态，导致很大的外推系数。可限制历史长度、删除相关向量、使用稳定求解或重启。具体阈值和回退策略应记录，并检查外推后的真实残差。', 'book:10.6.2'),
('Is a zero gradient sufficient to prove an orbital solution is a minimum?', 'No. It establishes stationarity within the allowed directions. Examine the relevant Hessian or stability problem to distinguish a local minimum from a saddle point. A local minimum also need not be the global minimum.', 'book:10.8'),
('Why compare finite differences over several step sizes?', 'A large displacement has truncation error, whereas a very small displacement amplifies rounding and incomplete inner convergence. A range of step sizes can reveal a stable agreement window. Use the same objective and conventions on both sides of the comparison.', 'protocol:original'),
('方法论文的耗时表应该包含什么？', '给出硬件、线程数、软件版本、初始轨道、停止阈值、体系与基组，并报告总耗时、主要构造次数和失败情况。对不同物理模型还需单独比较精度。没有实测结果时，只能描述测试方案。', 'protocol:original'),
('Can a published method be cited from a filename alone?', 'No. Verify title, authors, year and persistent identifier against the paper or a primary bibliographic record. Filenames can use preprint years or informal labels. If metadata is unresolved, mark it as unresolved rather than inventing a DOI.', 'protocol:original'),
('若代码能运行，但能量与参考不同，应先检查什么？', '先统一单位、几何、基组、电荷、自旋、冻结轨道、积分与密度约定、活性空间及核排斥项，再检查收敛和根跟踪。能运行只说明未触发程序错误，不证明实现了同一数学问题。', 'protocol:original'),
('What does PySCF mcscf.CASSCF(mf, ncas, nelecas) specify?', 'The arguments are the reference mean-field object, the number of active spatial orbitals, and the active electron count or an alpha/beta pair. ncas is not the electron count. Specify orbital selection and spin treatment separately when needed.', 'pyscf:mcscf_user'),
('PySCF sort_mo 默认轨道编号从0还是1开始？', '常用 mc.sort_mo 的 caslst 默认采用1起始轨道编号。可通过接口参数改变约定时，应明确设置并核对版本。NumPy切片一般是0起始，不能混用这两种编号。', 'pyscf:mcscf_user'),
('What does a state-averaged CASSCF energy represent?', 'It represents the weighted objective, not automatically the ground-state energy. Inspect the state-resolved energies when comparing roots or excitation gaps. Do not report the weighted objective as the energy of one physical state.', 'pyscf:mcscf_user'),
('PySCF 的 frozen 轨道与活性电子数有什么区别？', 'frozen 控制哪些轨道在轨道优化中不旋转，活性电子数定义 CI 问题的电子扇区。这是不同控制量；还要区分 MCSCF 轨道冻结参数与具体 CI 求解器的参数含义。', 'pyscf:mcscf_user'),
('Write a concise methods paragraph for an RHF calculation without inventing numerical results.', 'We solve the restricted Hartree-Fock equations in the specified atomic-orbital basis, iterating the spin-summed density and Fock matrix to self-consistency. The total energy includes nuclear repulsion. Reproduction requires the molecular geometry, units, basis, charge, convergence thresholds and implementation version; these values must be supplied before the paragraph is final.', 'book:10.6'),
('Write a short algorithm description for fixed-weight state-averaged CASSCF.', 'Choose an active space and normalized nonnegative state weights. For the current orbitals, solve for the requested active-space states and form the weighted objective. Update the allowed orbital rotations using the chosen optimizer and repeat until the CI, orbital and energy criteria are satisfied. Report individual-state energies separately from the averaged objective.', 'book:12.7.2'),
('A paper claims an 8x speedup. May I reuse that number for my implementation?', 'No. A published speedup belongs to its stated systems, baseline, hardware and tolerances. Cite it with those qualifications if relevant. Measure your implementation under controlled conditions before making its own performance claim.', 'protocol:original'),
('CIS 与 FCI 是同一种截断吗？', '不是。CIS 相对于指定参考只保留单激发相关的组态空间；FCI 在固定单粒子基与电子数、自旋投影等设定下包含全部相应行列式。二者的变分空间和成本不同，不能把 CIS 结果称为该基中的精确结果。', 'book:11'),
]
for i, (question, answer, source) in enumerate(concepts):
    add(f'C{i+1:03d}', question, answer, [source])

# Computed exercises: train/validation are separated by explicit parameters.
# These are bounded-transfer tests, not independent source-level benchmarks.
for n in (2,3,5,6,8,10):
    for holes in (False,True):
        electrons=2*n-1 if holes else 1
        add(f'N{n:02d}{int(holes)}',
            f'非相对论且核闭壳层，CAS({electrons},{n}) 中有几个未成对电子，允许什么自旋？',
            f'该活性空间是{"单空穴" if holes else "单电子"}空间，只有一个未成对电子。S=1/2，2S+1=2，为双重态；闭壳层核不改变总自旋。',
            ['book:2.4'],split='valid' if n==10 else 'train',topic='electron_count')

for n,na,nb in ((4,2,2),(5,2,1),(6,3,3),(7,3,2),(8,2,2),(9,4,3)):
    dim=math.comb(n,na)*math.comb(n,nb)
    add(f'D{n:02d}',f'{n}个空间轨道，Nalpha={na}、Nbeta={nb}。固定自旋投影行列式空间有多大？',
        f'分别选择 alpha 与 beta 占据，维数为 C({n},{na}) C({n},{nb})={dim}。这不是固定总自旋S的CSF数量；不能混同自旋投影扇区与自旋适配子空间。',
        ['book:11'],split='valid' if n==9 else 'train',topic='combinatorics')

for ncore,ncas,nvirt in ((1,3,4),(2,4,6),(3,5,7),(4,6,8),(2,3,5),(1,2,7)):
    count=ncore*ncas+ncore*nvirt+ncas*nvirt
    add(f'R{ncore}{ncas}{nvirt}',
        f'实轨道完整CAS，非活性/活性/虚轨道数为{ncore}/{ncas}/{nvirt}，无对称性或冻结限制。通常有几个非冗余轨道旋转参数？',
        f'计入非活性—活性、非活性—虚轨道、活性—虚轨道三类：{ncore}×{ncas}+{ncore}×{nvirt}+{ncas}×{nvirt}={count}。完整CAS的子空间内部旋转在相应CI变换下冗余。此计数不适用于任意截断或受约束模型。',
        ['book:12.2.6'],split='valid' if nvirt==8 else 'train',topic='rotation_count')

for s,ne in ((3,2),(4,2),(5,2),(2,1),(7,1),(6,2)):
    add(f'M{s}{ne}',f'单个归一化空间轨道的AO度量S={s}，系数C=1/sqrt({s})，占据数f={ne}。求D和Tr(DS)。',
        f'D=f CC†={ne}/{s}。Tr(DS)=({ne}/{s})×{s}={ne}，正确恢复电子数。标量例子中DSD=fD；自旋求和闭壳层f=2与单占据f=1的密度约定不同。',
        ['book:10.7'],split='valid' if s==7 else 'train',topic='ao_metric')

code_tasks=[
('用NumPy从轨道系数和占据数构造AO密度，允许复系数。', '''import numpy as np
def density_matrix(coeff, occupations):
    coeff = np.asarray(coeff)
    occupations = np.asarray(occupations)
    return (coeff * occupations) @ coeff.conj().T
列对应分子轨道。若occupations是自旋求和占据数，RHF占据列取2。电子数需用Tr(DS)检查，而非默认Tr(D)。''', 'book:10.7'),
('写NumPy函数检验MO在给定AO度量下的正交性。', '''import numpy as np
def orthogonality_error(coeff, overlap):
    gram = coeff.conj().T @ overlap @ coeff
    return np.linalg.norm(gram - np.eye(gram.shape[0]))
这是数值残差，不代替科学收敛判据。函数要求系数行与AO度量维度对应。''', 'book:3.1'),
('Write code for a CI residual using a matrix-vector product callback.', '''import numpy as np
def ci_residual(matvec, vector, energy):
    vector = np.asarray(vector)
    return np.linalg.norm(matvec(vector) - energy * vector)
Normalize the vector and state the residual tolerance. A small residual alone does not identify which root was found.''', 'book:11'),
('给出PySCF计算He原子RHF的最小输入，不报告虚构能量。', '''from pyscf import gto, scf
mol = gto.M(atom="He 0 0 0", basis="sto-3g", charge=0, spin=0)
mf = scf.RHF(mol)
mf.kernel()
if not mf.converged:
    raise RuntimeError("SCF did not converge")
print(mf.e_tot)
能量由实际运行返回；此代码没有预先假定其数值。''', 'pyscf:hf.py:kernel'),
('How can I obtain the AO density and overlap from a converged PySCF RHF object mf?', '''import numpy as np
dm = mf.make_rdm1()
overlap = mf.get_ovlp()
electron_count = np.trace(dm @ overlap).real
print(electron_count)
For RHF, make_rdm1 returns the spin-summed density. UHF returns spin-resolved components, so its shape and summation need separate handling.''', 'pyscf:hf.py:kernel'),
('在PySCF中计算Li原子的ROHF双重态应怎样设置spin？', '''from pyscf import gto, scf
mol = gto.M(atom="Li 0 0 0", basis="sto-3g", charge=0, spin=1)
mf = scf.ROHF(mol).run()
if not mf.converged:
    raise RuntimeError("ROHF did not converge")
print(mf.e_tot)
spin=1设置Nalpha-Nbeta=1，双重态多重度为2；不要把spin设成多重度2。''', 'pyscf:mole.py:2185'),
('Write a NumPy function computing the real RHF energy from spin-summed D, h, J and K.', '''import numpy as np
def rhf_energy(d, h, j, k, nuclear_repulsion):
    f = h + j - 0.5 * k
    return 0.5 * np.trace(d @ (h + f)).real + nuclear_repulsion
Here J and K must be built from the same spin-summed density convention. This formula is not a universal UHF or correlated energy expression.''', 'book:10.7'),
('用NumPy写中心差分方向导数的辅助函数。', '''import numpy as np
def central_directional_derivative(energy, x, direction, step):
    x = np.asarray(x)
    direction = np.asarray(direction)
    if step <= 0:
        raise ValueError("step must be positive")
    return (energy(x + step * direction) - energy(x - step * direction)) / (2 * step)
比较时使用同一目标与相同内层收敛精度，并扫描多个step。''', 'protocol:original'),
]
for i,(q,a,ref) in enumerate(code_tasks): add(f'P{i+1:03d}',q,a,[ref],topic='code')

extra_valid=[
('What is wrong with treating a stationary saddle point as the global ground state?', 'A vanishing gradient only establishes stationarity. Check stability to the allowed orbital variations and compare relevant solutions; neither stationarity nor local stability establishes the global minimum.', 'book:10.8'),
('Can two calculations with different active spaces be compared using iteration counts alone?', 'No. They solve different variational problems and may have different work per iteration and accuracy. Compare wall time, major operations, convergence success, final-state identity and physical accuracy under a clearly specified protocol.', 'protocol:original'),
('如何区分CI系数优化和轨道优化？', '前者在给定轨道基上改变组态线性组合，后者改变构造组态的单粒子轨道。CASSCF同时涉及二者，算法中的变量消元或内外循环划分不改变这一区别。', 'book:12.1'),
]
for i,(q,a,ref) in enumerate(extra_valid): add(f'W{i+1:03d}',q,a,[ref],split='valid')

source_ids={s['id'] for s in json.loads((ROOT/'sources.json').read_text())}
assert all(ref in source_ids for r in records for ref in r['source_refs'])
seen=set()
for r in records:
    prompt=r['messages'][1]['content']
    assert prompt not in seen, prompt
    seen.add(prompt)
for split in ('train','valid'):
    rows=[r for r in records if r['split']==split]
    (ROOT/'data'/f'{split}.jsonl').write_text(''.join(json.dumps({'messages':r['messages']},ensure_ascii=False)+'\n' for r in rows))
    print(split, len(rows))
(ROOT/'data/provenance.json').write_text(json.dumps([{k:v for k,v in r.items() if k!='messages'} for r in records],ensure_ascii=False,indent=2)+'\n')
