"""Ghép cấu trúc, mật độ, năng lượng PBE và nhãn tham chiếu thành HDF5 (06_build_dataset.ipynb)."""
import json
import os

import h5py
import numpy as np
import pandas as pd

from . import density_utils as du

UNITS = {
    "energy": "Ha (Hartree); cột *_eV_per_atom đổi bằng 1 Ha = 27.211386 eV",
    "length": "lattice_A, frac_coords: Å / phân số; rho: e/Bohr^3 trên lưới FFT đều",
    "rho": "mật độ electron hóa trị tổng (giả thế GTH), shape = mesh FFT",
    "delta_ref_pbe_Ha": "E_ref[rho_PBE] - E_PBE[rho_PBE], tính trên cùng ma trận mật độ",
}


def build_h5(h5_path, selected, density_dir, labels, settings, meta, rho_tol=1e-4):
    """Tạo HDF5; trả về manifest (mỗi vật liệu một dòng, kèm trạng thái và lý do loại nếu có)."""
    lab = labels.set_index("material_id")
    rows = []
    with h5py.File(h5_path, "w") as h5:
        h5.attrs["units"] = json.dumps(UNITS, ensure_ascii=False)
        h5.attrs["pbe_settings"] = json.dumps(settings, ensure_ascii=False)
        h5.attrs["meta"] = json.dumps(meta, ensure_ascii=False, default=str)
        grp = h5.create_group("materials")
        for r in selected.itertuples():
            mid = r.material_id
            row = {"material_id": mid, "split": r.split, "included": False, "reason": ""}
            npz = os.path.join(density_dir, f"{mid}.npz")
            js = os.path.join(density_dir, f"{mid}.json")
            if not (os.path.exists(npz) and os.path.exists(js)):
                row["reason"] = "thiếu file mật độ"
            elif mid not in lab.index or not np.isfinite(lab.loc[mid, "delta_ref_pbe_Ha"]):
                row["reason"] = "thiếu nhãn tham chiếu"
            else:
                info = json.load(open(js))
                d = du.load_density(npz)
                n_el = du.integrate(d["rho"], d["lattice_A"])
                if not info.get("converged"):
                    row["reason"] = "SCF không hội tụ"
                elif abs(n_el - info["nelec_valence"]) > rho_tol:
                    row["reason"] = f"∫ρ lệch {n_el - info['nelec_valence']:+.2e}"
                elif not np.all(np.isfinite(d["rho"])):
                    row["reason"] = "ρ chứa NaN/inf"
                else:
                    g = grp.create_group(mid)
                    g.create_dataset("rho", data=d["rho"].astype(np.float32), compression="gzip")
                    g.create_dataset("lattice_A", data=d["lattice_A"])
                    g.create_dataset("frac_coords", data=d["frac_coords"])
                    g.create_dataset("species", data=np.array([str(x) for x in d["species"]], dtype="S3"))
                    L = lab.loc[mid]
                    for k in ("e_pbe_Ha", "e_lda_Ha", "e_ref_Ha", "delta_ref_pbe_Ha"):
                        g.attrs[k] = float(L[k])
                    g.attrs.update(split=r.split, formula=str(r.formula_pretty), n_atoms=int(len(d["species"])),
                                   band_gap_MP_eV=float(r.band_gap_eV), converged=True,
                                   nelec_valence=float(info["nelec_valence"]),
                                   kmesh=np.array(info["kmesh"]), e_pbe_scf_Ha=float(info["e_tot_Ha"]))
                    row["included"] = True
            rows.append(row)
    return pd.DataFrame(rows)


def list_materials(h5_path, split=None):
    with h5py.File(h5_path, "r") as h5:
        return [m for m, g in h5["materials"].items() if split is None or g.attrs["split"] == split]


def load_sample(h5_path, mid):
    """Đọc một mẫu: dict gồm rho, lattice_A, species, nhãn và thuộc tính."""
    with h5py.File(h5_path, "r") as h5:
        g = h5["materials"][mid]
        out = {"material_id": mid, "rho": g["rho"][()].astype(np.float64), "lattice_A": g["lattice_A"][()],
               "frac_coords": g["frac_coords"][()], "species": [s.decode() for s in g["species"][()]]}
        out.update({k: (v.item() if hasattr(v, "item") and np.ndim(v) == 0 else v) for k, v in g.attrs.items()})
    return out


def read_targets(h5_path, split=None):
    """Đọc nhãn/thuộc tính vô hướng (không đọc mật độ) của một split; dùng cho baseline và thống kê."""
    rows = []
    with h5py.File(h5_path, "r") as h5:
        for mid, g in h5["materials"].items():
            if split is not None and g.attrs["split"] != split:
                continue
            n = int(g.attrs["n_atoms"])
            rows.append({"material_id": mid, "split": g.attrs["split"], "formula": g.attrs["formula"],
                         "n_atoms": n, "band_gap_MP_eV": float(g.attrs["band_gap_MP_eV"]),
                         "delta_ref_pbe_eV_per_atom": float(g.attrs["delta_ref_pbe_Ha"]) * 27.211386245988 / n,
                         "delta_lda_pbe_eV_per_atom": (float(g.attrs["e_lda_Ha"]) - float(g.attrs["e_pbe_Ha"]))
                         * 27.211386245988 / n})
    return pd.DataFrame(rows)
