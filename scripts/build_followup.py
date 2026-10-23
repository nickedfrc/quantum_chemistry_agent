"""Build a separate follow-up dataset without changing the completed pilot."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'experiments/followup'
DEST.mkdir(parents=True, exist_ok=True)
SYSTEM = (ROOT / 'system_prompt.txt').read_text().strip()
source = (ROOT / 'examples/verified_cases.py').read_text()
functions = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
results = {r['id']: r for r in json.loads((ROOT/'results/code-cases.json').read_text())['cases']}
rows, provenance = [], []


def add(identifier, split, family, prompt, answer, refs):
    rows.append(dict(split=split, messages=[dict(role='system',content=SYSTEM),
        dict(role='user',content=prompt),dict(role='assistant',content=answer)]))
    provenance.append(dict(id=identifier, split=split, family=family, source_refs=refs,
        review_status='Original AI-authored teaching example; cited APIs checked; not independent expert review'))


# Each family has distinct code, algorithm, numerical interpretation and repair tasks.
# The code and numerical outputs were executed before constructing these records.
families = [
('rhf_water', '水分子的 RHF 能量与自洽残差', 'train', 'pyscf:scf_user',
 'Build the molecule and solve RHF to the stated tolerance. Form the spin-summed AO density, overlap and core Hamiltonian. Build Coulomb and exchange from the same density, set F=h+J-K/2, and reconstruct the total energy including nuclear repulsion. Check the metric electron count and FDS-SDF residual. These checks test numerical consistency for this small calculation; they do not establish basis-set accuracy or global stability.',
 '电子数为10，能量重构与 PySCF 的差约为双精度舍入误差，自洽残差很小。这支持本例的密度、积分和能量约定一致。STO-3G 基组误差与 RHF 缺失的电子相关并没有因此消失，也没有完成稳定性分析。',
 '把自旋求和密度的 Fock 写成 h+2J-K 有什么问题？',
 '若 J、K 已由 D=2 Cocc Cocc† 构造，应使用 F=h+J-K/2。h+2J-K 适用于另一种未含双占据因子的空间密度约定，不能再配合自旋求和 D。修正后用总能量重构和 Tr(DS) 检查因子。'),
('mp2_hf', 'HF 分子的 MP2 相关能', 'train', 'pyscf:mp_user',
 'Converge the RHF reference for the specified geometry and basis. Construct mp.MP2(mf), solve for the perturbative double-excitation amplitudes, and distinguish correlation energy from total energy. Check Etotal=EHF+Ecorr and the trace of the MO one-particle density. This small-basis calculation is an API and convention check, not a high-accuracy molecular prediction.',
 '总能量等于 RHF 能量加 MP2 相关能；负的相关能在这个例子中降低了能量。默认 make_rdm1 返回 MO 表示，故正交 MO 中迹为10。不能把该迹检查推广成 MP2 密度一定满足所有可表象性条件，也不能把 MP2 称为一般变分上界。',
 'mp.MP2(mf).kernel() 的第一个返回量能直接写作总能量吗？',
 '不能。第一个返回量是 MP2 相关能，另一个是 t2。用 calc.e_tot，或经核对后用 mf.e_tot+e_corr 表示总能量。不要把相关能再次加到 calc.e_tot 上。'),
('ccsd_h2', 'H2 的 CCSD 振幅与能量', 'train', 'pyscf:cc_user',
 'First converge RHF, then construct cc.CCSD from that reference and iterate the singles and doubles equations. Require both SCF and CC convergence. Report the reference energy, correlation energy and their sum separately. Amplitude norms describe this solution but are not by themselves a general accuracy certificate. Record basis, geometry, thresholds and any frozen-orbital choice.',
 '这个二电子例子的 CCSD 已满足所设停止条件，总能量关系正确。t1、t2 范数是振幅信息，不是实际误差估计；不能仅从收敛或能量降低推断所有体系上可靠，也不能把本例成本当作大体系标度测试。',
 '只检查 mf.converged 就写 CCSD 已收敛是否足够？',
 '不够。mf.converged 只表示参考 SCF 已收敛，还须检查 calc.converged。若 CCSD 未收敛，应报告失败并检查振幅迭代、参考态和数值设置，不能悄悄把最后一个迭代能量当作已收敛结果。'),
('casci_lih', 'LiH 的固定轨道 CASCI', 'train', 'pyscf:mcscf_user',
 'Converge the RHF reference and select two active spatial orbitals with one alpha and one beta active electron. Solve the active-space CI problem at those fixed orbitals. Inspect the active one-particle density and total-spin expectation. LiH has four electrons here, so one doubly occupied inactive core orbital remains outside the two active electrons. State the orbital selection; an electron count alone does not define it.',
 '活性密度的迹为2，只计算活性电子；还存在一个双占据非活性核轨道，所以总电子数为4。S²接近0支持本次求得的单重态。两个自然占据接近2和0，不等于波函数完全没有相关，也不能单凭占据决定总自旋。',
 '为什么不能把活性 RDM 的迹2写成 LiH 的总电子数？',
 '活性 RDM 不包含双占据核轨道。总电子数是 2*ncore+sum(nelecas)=2*1+2=4。完整 AO 密度与活性空间密度有不同定义，报告时必须标注。'),
('casscf_lih', 'LiH 的 CASCI 与 CASSCF 比较', 'train', 'pyscf:mcscf_user',
 'Use the same molecular Hamiltonian, initial orbitals and active-electron sector for both calculations. CASCI varies only CI coefficients. CASSCF also varies the allowed orbital rotations, iterating the coupled CI and orbital problem. Check both convergence flags and compare total energies in Hartree. A lower CASSCF energy in this run is an orbital-relaxation observation, not proof that either calculation found a global minimum.',
 '在同一基组和电子扇区中，CASSCF 比固定 RHF 轨道的 CASCI 能量更低；差值约为 -0.0189164 Hartree，反映本次轨道优化带来的变化。该结果并不能证明全局最优、完整动态相关或普遍速度优势。',
 '把 CASSCF 的核轨道数 ncore 当作冻结轨道参数 frozen 是否正确？',
 '不正确。ncore 指非活性双占据空间的大小；这些轨道与其他空间之间的旋转通常仍属于轨道优化问题。frozen 才额外约束指定轨道不参与相应优化。二者不能互换。'),
('uhf_hydrogen', '氢原子 UHF 自旋检查', 'train', 'pyscf:scf_user',
 'For neutral atomic hydrogen set spin=1, meaning Nalpha-Nbeta=1. Converge UHF, obtain separate alpha and beta AO densities, and trace each against the AO overlap. Inspect S squared with spin_square. This one-electron example has S=1/2 and multiplicity two. The same spin projection would not alone establish spin purity for an arbitrary many-electron UHF solution.',
 'alpha 电子数约1、beta 为0，S²=0.75=S(S+1) 对应 S=1/2、多重度2。PySCF 的 spin=1 是电子数差，而不是多重度1，也不是 S=1。这里的纯净结果不能保证一般 UHF 行列式都自旋纯。',
 '对 UHF 的 make_rdm1 直接写 np.trace(dm @ S) 为什么危险？',
 'UHF 密度有 alpha、beta 两个分量；应先 da,db=mf.make_rdm1()，再分别计算 trace(da@S)、trace(db@S)，总电子数为两者之和。直接对三维数组使用 trace 的默认轴会混淆自旋轴与 AO 轴。'),
('integral_transform', 'AO 到 MO 双电子积分变换', 'train', 'pyscf:ao2mo',
 'Build real RHF orbitals, evaluate the full AO electron-repulsion tensor, and apply the four orbital coefficient contractions in chemists notation. Independently transform with ao2mo.kernel and restore the compact result to four indices. Compare tensor shapes and a maximum absolute error. The explicit four-index route is for a tiny validation example; its memory cost makes it inappropriate for large systems.',
 '完整 MO 积分张量形状为2×2×2×2，压缩表示形状为3×3，不能直接把两个数组逐元素比较。还原后误差约4e-16，支持此实轨道例子的指标顺序一致。复轨道需重新核对共轭与积分对称性，不能照搬实数公式。',
 'ao2mo.kernel 返回数组能不经检查就 reshape 成四指标张量吗？',
 '不能。默认可能采用对称压缩存储；本例应使用 ao2mo.restore(1, packed, nmo)。先确认积分记号、轨道是否实数、对称性与目标形状，避免把压缩索引误当完整 AO/MO 指标。'),
('generalized_eigenproblem', '广义厄米本征问题', 'train', 'book:10.6',
 'Verify that the overlap matrix is Hermitian positive definite. Solve F C=S C epsilon using a generalized Hermitian eigensolver. Check both the eigen-equation residual and C dagger S C-I. Eigenvector signs may differ between implementations; compare invariant subspaces or aligned vectors when necessary. Near-linear dependence requires an explicit threshold or reduced representation.',
 '两个残差都接近机器精度，说明本例广义本征方程与 AO 度量归一同时满足。只对 F 用 eigh 会解成另一个问题。残差小也不代表 F 已对某个电子密度自洽；这里仅求解固定矩阵。',
 '为什么不能用 np.linalg.eigh(F) 替代 scipy.linalg.eigh(F,S)？',
 '前者假定欧氏度量 I，后者求解 FC=SCε。S≠I 时二者一般不同。可以先构造合适正交化变换再求普通本征问题，但还须将系数变回 AO 表示并检查 C†SC=I。'),
('diis_coefficients', 'DIIS 受约束最小二乘系数', 'train', 'paper:pulay1982',
 'Collect compatible residual vectors and form their Gram matrix. Minimize the squared norm of their linear combination subject to coefficients summing to one. Solve the augmented linear system, then extrapolate the corresponding stored iterates. Inspect conditioning and the actual nonlinear residual after extrapolation. This synthetic coefficient calculation illustrates the constraint; it does not demonstrate convergence of an SCF calculation.',
 '系数之和为1，外推误差范数约0.857143。此构造中系数恰为正并不意味着 DIIS 系数总为正。线性组合的误差变小，也不保证下一次非线性 SCF 能量或残差单调下降。',
 'DIIS Gram 矩阵近奇异时，给系数加任意小数再归一是否合理？',
 '这不是有依据的通用修复。先诊断历史残差的线性相关，可删去相关向量、缩短历史或重启；采用稳定线性求解与明确的正则化策略时应记录参数。修复后仍检查真实 SCF 残差。'),
('rayleigh_ritz', 'Rayleigh–Ritz 子空间残差', 'valid', 'paper:davidson1975',
 'Orthonormalize the trial subspace, project the Hermitian Hamiltonian, solve its small eigenproblem and lift the Ritz vector to the full space. Check the full residual as well as its subspace projection. The projected residual vanishes by construction, while the full residual can remain large. Expand or improve the subspace before declaring an eigenvector converged.',
 '投影残差接近0但全空间残差约1.337，因此它是子空间驻定解，不是已收敛的全空间本征对。Ritz 能量高于完整矩阵的最低本征值符合变分界，但不表示误差已经很小。',
 'Q.T @ (Hc-Ec) 接近零能否作为 Davidson 求解器唯一停止条件？',
 '不能。该投影量在准确解出子空间本征问题后本来就应接近零。应检查完整残差 Hc-Ec 的范数、目标根身份及正交性；必要时预条件残差并扩充子空间。'),
]

imports = 'import numpy as np\nfrom scipy.linalg import eigh\nfrom pyscf import gto, scf, mp, cc, mcscf, ao2mo\n'
for name, title, split, ref, algorithm, interpretation, bug, fix in families:
    assert results[name]['checks'] == 'passed'
    code = imports + '\n' + functions[name] + '\n\nprint(' + name + '())'
    add(name+'-code', split, name, f'写一个可运行的 Python 小例子，演示{title}，明确设置并加入必要数值检查。',
        '以下是已在 PySCF 2.6.2 上运行通过的教学例子；具体结果由运行返回。\n```python\n'+code+'\n```', [ref])
    add(name+'-method', split, name, f'Write a methods or algorithm paragraph for this calculation without inventing results:\n```python\n{functions[name]}\n```', algorithm, [ref])
    add(name+'-analysis', split, name, f'分析{title}的实际输出，并说明它不能证明什么：\n'+json.dumps(results[name]['result'],ensure_ascii=False), interpretation, [ref])
    add(name+'-repair', split, name, bug, fix, [ref])

concepts = [
('mp2_denominator','MP2 在近简并问题中为什么可能不可靠？','MP2 以单参考微扰展开为基础。典型双激发能量分母为 εi+εj−εa−εb；小分母可能使振幅变大，使低阶展开不再受控。应检查参考态、轨道能隙和问题的多参考性质，不能只因计算顺利结束就相信结果，也不能把某个阈值当作普遍定理。','pyscf:mp_user'),
('mp2_total','写一个说明 MP2 总能量与相关能区别的方法段落。','我们先在指定基组和分子几何下收敛 Hartree–Fock 参考态，再计算二阶微扰相关能。所报总能量为参考能与相关修正之和。需要另外列明自旋参考、冻结轨道、积分处理和程序版本；本文段不预设数值精度，也不宣称 MP2 为变分方法。','pyscf:mp_user'),
('mp2_rdm','为什么相关方法的一粒子密度一般不能继续用 2CoccCocc†？','2CoccCocc† 是闭壳层单行列式的自旋求和密度。相关波函数通常改变自然轨道占据，应使用相应方法定义的 RDM；还要区分 MO/AO 表示与松弛/非松弛密度。只把 HF 占据矩阵改名为相关密度，会遗漏相关贡献。','pyscf:mp_user'),
('cc_ansatz','解释 CCSD 的指数波函数与截断的含义。','标准单参考 CCSD 写作 |Ψ>=exp(T1+T2)|Φ0>，T1、T2 分别含单、双激发算符。截断的是连通簇算符；指数展开仍会生成更高激发阶数的非连通乘积。因此不能说 CCSD 波函数只含单、双激发行列式，也不能把它与 CISD 等同。','pyscf:cc_user'),
('cc_variational','CCSD 能量比某参考计算更低，就能判断它更准确吗？','不能。标准投影 CCSD 并非 Rayleigh–Ritz 变分最小化，其能量没有一般的严格上界保证。应先统一哈密顿量、基组和电子扇区，再与适当基准及收敛信息比较；能量更低不自动意味着误差更小。','pyscf:cc_user'),
('cc_lambda','CCSD 的 Lambda 方程有什么作用？','投影 CC 理论通常是非变分、非厄米的表示。Lambda 振幅构造相应左态或拉格朗日乘子，参与密度与响应性质的计算。它们不是简单把右态振幅复共轭得到；求性质时应核对具体方法是否已求解所需 Lambda 方程。','pyscf:cc_user'),
('cc_triples','PySCF 中 ccsd_t() 返回的值应该怎么加入能量？','它返回扰动三激发修正。若 calc 已完成 CCSD，则 E(CCSD(T))=calc.e_tot+calc.ccsd_t()；不要把返回值当完整总能量，也不要再次加已包含在 calc.e_tot 中的 CCSD 相关能。需要保证参考和 CCSD 解适用于所选处理。','pyscf:cc_user'),
('cc_size','区分尺寸广延性与逐步提高基组的收敛性。','尺寸广延性讨论非相互作用片段组合时能量是否按片段相加；基组收敛性讨论扩大单粒子空间时结果如何趋近极限。二者是不同性质。某方法尺寸广延，不能据此宣称小基组已收敛，或对所有强相关体系准确。','pyscf:cc_user'),
('frozen_core','MP2/CCSD 中 frozen=1 与电子数减少2是否是一回事？','不是同一件事。冻结一个双占据核轨道通常是限制相关激发空间，并不把这两个电子从分子总电子数与参考平均场中删除。应记录冻结空间和能量约定，不能把冻结核相关近似当作改变体系电荷。','pyscf:cc_user'),
('basis_limit','为什么 STO-3G 的计算适合程序检查，却不能直接作为高精度方法验证？','小基组降低运行成本，适合核对接口、维度、电子数、符号和能量重构。它的基组不完备误差可能很大。若要验证化学精度，需要另外检查基组收敛、电子相关处理和参考数据；程序检查成功不等于科学误差已小。','protocol:original'),
('ao_hcore','用 PySCF 分解非相对论全电子核心哈密顿量，并说明适用边界。','对于常规非相对论全电子分子，hcore 包含动能与电子—核吸引项，可检查 mol.intor("int1e_kin")+mol.intor("int1e_nuc") 与 mf.get_hcore()。若加入 ECP、相对论或外场修正，不能仍假设这两个积分已包含所有一电子项。','paper:libcint2015'),
('integral_notation','双电子积分 (pq|rs) 的指标顺序为什么必须先说明？','化学家与物理学家记号使用不同的指标分组方式。代码中张量轴排列、交换项及振幅收缩都依赖约定。不能仅因数组形状相同就直接复用 einsum；应写出积分定义，再以小体系显式收缩和对称性检查验证。','pyscf:ao2mo'),
('orbital_phase','两个程序给的分子轨道正负号相反，说明一个错了吗？','不一定。单个本征矢可任意乘以相位，实轨道常表现为符号翻转；简并空间还可作酉旋转。先比较密度、能量、残差和子空间，再在需要时作相位或子空间对齐。不要直接以系数逐元素相减判断物理不一致。','book:3.1'),
('scf_steps','给出可以写进方法部分的 SCF 停止条件说明。','明确所用能量变化、密度变化或轨道梯度/残差阈值，以及最大迭代次数。报告实际是否满足这些条件；若只有能量变化小而残差仍大，不能称为可靠自洽。所设阈值应由具体计算记录填写，不能替作者编造。','pyscf:scf_user'),
('iteration_plateau','SCF 能量连续几步没变，但残差仍大，应如何分析？','先核对输出精度和真实能量变化，再查看密度/轨道残差。平台可能来自舍入显示、停滞、外推病态或受限步长；仅看能量小变化不能区分这些原因。检查历史误差与条件数，必要时调整初猜或求解策略，并保留失败记录。','pyscf:scf_user'),
('energy_components','结果表为什么要分别写电子能和核排斥能？','电子哈密顿量本征值或平均值不总是包含核排斥常数，而分子总能量通常包含。不同接口若混用会产生看似巨大的差异。先核对 Etotal=Eelectronic+Enuclear 的约定，再比较方法误差或势能曲线。','book:10.7'),
('cas_orbital_choice','相同 CAS(2,2) 标签是否足以复现一次计算？','不足。还需指定哪两个空间轨道进入活性空间、初始轨道、分子与基组、自旋扇区、对称性、目标根及停止条件。不同轨道选择可能描述不同相关过程；电子/轨道数相同不保证优化到同一个解。','pyscf:mcscf_user'),
('cas_natorb','怎样报告自然轨道占据而不夸大结论？','说明 RDM 的定义和所在空间，给出其本征占据以及电子数检查。占据明显偏离0或2可提示单行列式图像不足，但具体相关机制、自旋耦合和激发态归属还需其他波函数信息支持。不要仅凭占据阈值宣称某个唯一物理结论。','pyscf:mcscf_user'),
('code_method','从可运行代码生成方法段落，应优先提取哪些信息？','提取哈密顿量、几何与单位、基组、电荷/自旋、参考态、相关方法、活性或冻结空间、停止条件和程序版本。把默认值与显式设置区分开；计算输出支持的结果才能写成已发生事实。仅看函数名称不能补齐未设置的科学细节。','paper:pyscf2020'),
('method_result','怎样区分方法描述与结果分析？','方法描述给出可重复执行的定义、设置和步骤；结果分析解释实际输出、对照和不确定性。例如“解 CCSD 振幅方程”属于方法，“本次相关能为某数值”必须来自运行记录。没有记录时应提供待填写项，不能用看似合理的数值补空。','protocol:original'),
('validation_ritz','一个 Ritz 能量在迭代中下降，但完整残差不降，应该怎样判断？','能量下降只反映当前子空间的变分改善，不能替代本征残差检查。确认子空间正交性、数值精度、目标根与残差构造，再检查扩空间方向是否有效；避免仅以能量差作为唯一停止条件。','paper:davidson1975'),
('validation_root','为什么本征矢残差很小仍需检查目标根身份？','任何精确本征对的残差都为零，因此小残差不能区分基态、激发态或相邻根。结合能量排序、对称性、波函数重叠和物理性质跟踪目标态；简并附近最好检查整个相关子空间。','book:11'),
('validation_time','同一代码首次运行10秒、第二次3秒，可以直接写三倍加速吗？','不能据此归因于算法改变。首次运行可能包含初始化、缓存、库加载等额外工作。先固定输入、线程、停止条件和计时范围，区分冷启动与重复运行并做多次测量；结果中说明变异与失败情况。','protocol:original'),
('validation_publication','公开软件文档中的一个例子能否证明我的方法具有创新性？','不能。例子可以帮助核对接口和已有方法行为；创新性需根据明确的技术差异和适当的文献比较论证。不要将复现、参数改变或程序包装自动称为新算法，也不要编造尚未进行的对照结果。','protocol:original'),
]
for key, question, answer, ref in concepts:
    add('text-'+key, 'valid' if key.startswith('validation_') else 'train', key, question, answer, [ref])

original_prov = json.loads((ROOT/'data/provenance.json').read_text())
combined_prov = []
for split in ('train', 'valid'):
    old = [json.loads(line) for line in (ROOT/'data'/f'{split}.jsonl').read_text().splitlines()]
    new = [dict(messages=row['messages']) for row in rows if row['split']==split]
    (DEST/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in old+new))
    combined_prov.extend([r for r in original_prov if r['split']==split])
    combined_prov.extend([r for r in provenance if r['split']==split])
    print(split, len(old)+len(new), 'new', len(new))
(DEST/'provenance.json').write_text(json.dumps(combined_prov,ensure_ascii=False,indent=2)+'\n')
metadata = dict(status='validated dataset; see results/followup-01 for recorded experiment state', new_records=len(rows),
    numerical_cases_run=12, training_code_families=[f[0] for f in families if f[2]=='train'],
    validation_code_families=['rayleigh_ritz'], held_out_code_families=['gradient_check','density_fitting_check'],
    limitations='AI-authored and reviewed, not a blind or independently expert-adjudicated benchmark. Original pilot examples remain included; broad topics can overlap.',
    references_added=9, reference_access='Some paper references are metadata/abstract only; see sources.json.')
(DEST/'dataset_card.json').write_text(json.dumps(metadata,indent=2)+'\n')
