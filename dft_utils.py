"""Tiện ích DFT tuần hoàn bằng PySCF PBC (GTH pseudopotential + basis Gaussian).

Đơn vị: năng lượng Hartree (Ha), chiều dài Bohr trong PySCF; cấu trúc đầu vào là pymatgen (Å).
"""
import math
import os
import subprocess
import time

import numpy as np

HA2EV = 27.211386245988
MAGNETIC = {"Mn", "Fe", "Co", "Ni", "Cr", "V", "Gd"}   # nguyên tố từ tính: cần kiểm tra spin riêng

DEFAULT_SETTINGS = dict(
    basis="gth-dzvp-molopt-sr",
    pseudo="gth-pbe",
    xc="pbe",
    kspacing=0.30,          # 1/Å, quy ước VASP (gồm 2π)
    precision=1e-8,         # quyết định mesh FFT / cutoff
    exp_to_discard=0.1,
    spin=False,             # False: KRKS, True: KUKS
    conv_tol=1e-9,          # Ha
    conv_tol_grad=1e-6,
    max_cycle=200,
    smearing_ha=0.005,      # Fermi-Dirac cho kim loại / gap nhỏ
    gap_insul_eV=0.5,       # gap MP >= ngưỡng này -> occupations cố định
    max_memory_mb=None,
)


# ---------------------------------------------------------------- tài nguyên
def resources():
    """Báo cáo tài nguyên cho PySCF (CPU, RAM, luồng) và GPU (chủ yếu để huấn luyện JAX)."""
    info = {"cpu_count": os.cpu_count(), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")}
    try:
        info["ram_GB"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
    except (ValueError, OSError):
        info["ram_GB"] = None
    try:
        import pyscf
        from pyscf import lib
        info["pyscf"] = pyscf.__version__
        info["pyscf_threads"] = lib.num_threads()
    except Exception as e:  # pragma: no cover
        info["pyscf"] = f"lỗi: {e!r}"
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=10)
        info["gpu"] = out.stdout.strip() or None
    except Exception:
        info["gpu"] = None
    return info


# ---------------------------------------------------------------- dữ liệu basis/giả thế
def has_data(el, basis, pseudo):
    from pyscf.pbc import gto as pgto
    from pyscf.pbc.gto import pseudo as ppseudo
    try:
        return bool(pgto.basis.load(basis, el)) and bool(ppseudo.load(pseudo, el))
    except Exception:
        return False


def available_elements(elements, basis=DEFAULT_SETTINGS["basis"], pseudo=DEFAULT_SETTINGS["pseudo"]):
    return {e for e in set(elements) if has_data(e, basis, pseudo)}


# ---------------------------------------------------------------- cell và k-point
def build_cell(struct, s):
    from pyscf.pbc import gto as pgto
    cell = pgto.Cell()
    cell.a = struct.lattice.matrix
    cell.atom = [(site.specie.symbol, site.coords) for site in struct]
    cell.unit = "A"
    cell.basis = s["basis"]
    cell.pseudo = s["pseudo"]
    cell.precision = s["precision"]
    cell.exp_to_discard = s["exp_to_discard"]
    if s.get("max_memory_mb"):
        cell.max_memory = s["max_memory_mb"]
    cell.verbose = 0
    cell.build()
    return cell


def is_metallic(gap_eV, s):
    return (gap_eV is None) or (not np.isfinite(gap_eV)) or gap_eV < s["gap_insul_eV"]


def kmesh_from(struct, kspacing, even=False):
    b = struct.lattice.reciprocal_lattice_crystallographic.abc    # |b_i| không có 2π
    km = [max(1, math.ceil(2 * math.pi * x / kspacing)) for x in b]
    return [k + (k % 2) for k in km] if even else km


# ---------------------------------------------------------------- SCF
def run_scf(struct, gap_eV, settings=None):
    """Chạy SCF tuần hoàn. Trả về (info, rho, dm, kpts, cell).

    rho: mật độ electron tổng (e/Bohr^3) trên lưới FFT đều, dạng mảng 3D.
    dm: ma trận mật độ theo k (dùng để tính năng lượng không tự hợp ở bước 05).
    Với hệ smearing, lưới k lẻ từng cho sai số số electron khi thử nghiệm (Al), nên làm tròn lên chẵn.
    """
    from pyscf.pbc import dft as pdft, scf as pscf

    s = {**DEFAULT_SETTINGS, **(settings or {})}
    cell = build_cell(struct, s)
    metallic = is_metallic(gap_eV, s)
    km = kmesh_from(struct, s["kspacing"], even=metallic)
    kpts = cell.make_kpts(km)

    mf = (pdft.KUKS if s["spin"] else pdft.KRKS)(cell, kpts)
    mf.xc = s["xc"]
    mf.conv_tol, mf.conv_tol_grad, mf.max_cycle = s["conv_tol"], s["conv_tol_grad"], s["max_cycle"]
    if metallic:
        mf = pscf.addons.smearing_(mf, sigma=s["smearing_ha"], method="fermi")

    t0 = time.time()
    mf.kernel()
    attempts = 1
    if not mf.converged:                      # thử lại: level shift và nhiều vòng hơn
        attempts = 2
        mf.level_shift, mf.max_cycle = 0.3, 2 * s["max_cycle"]
        mf.kernel(dm0=mf.make_rdm1())
    wall = time.time() - t0

    rho = np.asarray(mf.get_rho())
    if s["spin"] and rho.ndim == 2:
        rho = rho.sum(axis=0)
    mesh = [int(m) for m in cell.mesh]
    n_el = float(rho.sum() * cell.vol / rho.size)
    rho = rho.reshape(mesh)

    info = dict(
        converged=bool(mf.converged), attempts=attempts, cycles=int(getattr(mf, "cycles", -1)),
        e_tot_Ha=float(mf.e_tot), e_tot_per_atom_Ha=float(mf.e_tot) / len(struct),
        nelec_valence=float(cell.nelectron), rho_integral=n_el, rho_err=n_el - float(cell.nelectron),
        electron_ok=bool(abs(n_el - cell.nelectron) < 1e-4),
        kmesh=km, nk=len(kpts), mesh=mesh, nao=int(cell.nao), wall_s=round(wall, 1),
        smearing=bool(metallic), xc=s["xc"], spin=bool(s["spin"]),
        e_free_Ha=float(mf.e_free) if metallic else None,
        e_zero_Ha=float(mf.e_zero) if metallic else None,
        entropy=float(mf.entropy) if metallic else None,
    )
    if hasattr(mf, "scf_summary"):
        info["scf_summary"] = {k: float(v) for k, v in mf.scf_summary.items() if np.isscalar(v)}
    return info, rho, np.asarray(mf.make_rdm1()), kpts, cell


def energy_on_density(cell, kpts, dm, xc, spin=False):
    """Năng lượng tổng (Ha) của functional `xc` đánh giá trên ma trận mật độ `dm` cho trước (không SCF).

    Không gồm đóng góp entropy của smearing; dùng cho PBE, LDA và functional tham chiếu trên cùng dm
    để hiệu năng lượng chỉ phản ánh khác biệt XC.
    """
    from pyscf.pbc import dft as pdft
    m = (pdft.KUKS if spin else pdft.KRKS)(cell, kpts)
    m.xc = xc
    return float(m.energy_tot(dm=dm))


def cell_from_arrays(lattice, species, frac_coords, settings):
    """Dựng lại cell PySCF từ dữ liệu lưu trong file mật độ."""
    from pymatgen.core import Lattice, Structure
    struct = Structure(Lattice(lattice), list(species), frac_coords)
    s = {**DEFAULT_SETTINGS, **settings}
    return struct, build_cell(struct, s)
