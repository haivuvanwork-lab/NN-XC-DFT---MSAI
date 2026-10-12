"""Tiện ích mật độ electron trên lưới FFT đều (đơn vị nguyên tử: e/Bohr^3, Bohr)."""
import json

import numpy as np

ANG2BOHR = 1.0 / 0.529177210903


def save_density(path, rho, lattice_A, species, frac_coords, kpts, dm_kpts, e_tot_Ha):
    """Lưu mật độ + dữ liệu cần để dựng lại cell và tính năng lượng không tự hợp."""
    np.savez_compressed(
        path, rho=np.asarray(rho, dtype=np.float64), lattice_A=np.asarray(lattice_A),
        species=np.array(species), frac_coords=np.asarray(frac_coords),
        kpts=np.asarray(kpts), dm_kpts=np.asarray(dm_kpts), e_tot_Ha=float(e_tot_Ha))


def load_density(path):
    d = np.load(path, allow_pickle=False)
    return {k: d[k] for k in d.files}


def integrate(rho, lattice_A):
    """∫ρ dr trên ô đơn vị (số electron). lattice_A: ma trận 3x3 theo Å."""
    vol_bohr3 = abs(np.linalg.det(np.asarray(lattice_A))) * ANG2BOHR ** 3
    return float(rho.sum() * vol_bohr3 / rho.size)


def _G_vectors(lattice_A, mesh):
    """Vector sóng G (Bohr^-1) cho lưới FFT, shape (n1,n2,n3,3)."""
    L = np.asarray(lattice_A) * ANG2BOHR            # hàng a_i, Bohr
    B = np.linalg.inv(L).T                          # hàng b_j, a_i·b_j = δ_ij (không có 2π)
    m = [np.fft.fftfreq(n) * n for n in mesh]
    M = np.stack(np.meshgrid(*m, indexing="ij"), axis=-1)
    return 2 * np.pi * M @ B


def gradient_laplacian(rho, lattice_A):
    """|∇ρ| và ∇²ρ bằng FFT (ρ tuần hoàn). Trả về (grad_norm, lap), cùng shape với rho."""
    G = _G_vectors(lattice_A, rho.shape)
    rho_g = np.fft.fftn(rho)
    grad = np.stack([np.fft.ifftn(1j * G[..., i] * rho_g).real for i in range(3)], axis=-1)
    lap = np.fft.ifftn(-(G ** 2).sum(-1) * rho_g).real
    return np.linalg.norm(grad, axis=-1), lap


def ml_features(rho, lattice_A, rho_floor=1e-10, s_max=20.0):
    """Đặc trưng cho ML-XC: ρ, s (gradient rút gọn), trọng số tích phân.

    s = |∇ρ| / (2 (3π²)^(1/3) ρ^(4/3)); ρ âm nhỏ do FFT/lưới được cắt về rho_floor.
    Trả về dict float32 phẳng: rho, s, w (= thể tích ô / số điểm lưới, Bohr^3).
    """
    gnorm, _ = gradient_laplacian(rho, lattice_A)
    r = np.maximum(rho, rho_floor)
    kf3 = (3 * np.pi ** 2) ** (1 / 3)
    s = np.clip(gnorm / (2 * kf3 * r ** (4 / 3)), 0.0, s_max)
    vol = abs(np.linalg.det(np.asarray(lattice_A))) * ANG2BOHR ** 3
    return {"rho": r.ravel().astype(np.float32), "s": s.ravel().astype(np.float32),
            "w": float(vol / rho.size)}


def dump_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=float)
