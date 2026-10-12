"""Kiểm tra cấu trúc, loại trùng, chọn mẫu và chia train/validation/test (02_preprocess_mp.ipynb)."""
import numpy as np
import pandas as pd


def read_structure(path):
    from pymatgen.core import Structure
    return Structure.from_file(path, primitive=False)


def min_distance(struct, r=5.0):
    """Khoảng cách nguyên tử-nguyên tử nhỏ nhất, tính cả ảnh tuần hoàn (Å)."""
    _, _, _, dists = struct.get_neighbor_list(r=r)
    return float(np.min(dists)) if len(dists) else float("inf")


def check_structure(struct, max_atoms=20, min_dist_A=0.7, max_vol_per_atom=200.0):
    """Trả về dict cờ kiểm tra; `ok` = True nếu đạt tất cả."""
    flags = {
        "ordered": bool(struct.is_ordered),
        "occupancy_ok": True,
        "n_atoms": len(struct),
        "volume_A3": float(struct.volume),
    }
    if struct.is_ordered:
        flags["occupancy_ok"] = all(abs(sum(site.species.values()) - 1.0) < 1e-6 for site in struct)
    flags["min_dist_A"] = min_distance(struct) if struct.is_ordered else float("nan")
    flags["cell_ok"] = bool(struct.volume > 1e-3 and np.all(np.isfinite(struct.lattice.matrix)))
    flags["vol_per_atom"] = flags["volume_A3"] / max(len(struct), 1)
    flags["ok"] = bool(
        flags["ordered"] and flags["occupancy_ok"] and flags["cell_ok"]
        and flags["n_atoms"] <= max_atoms
        and flags["min_dist_A"] >= min_dist_A
        and flags["vol_per_atom"] <= max_vol_per_atom)
    return flags


def dedupe_structures(df, structures, ltol=0.2, stol=0.3, angle_tol=5.0):
    """Loại cấu trúc trùng (cùng công thức rút gọn và StructureMatcher khớp).

    df: DataFrame có cột material_id, energy_above_hull_eV_per_atom.
    structures: dict material_id -> Structure.
    Trả về (danh sách material_id giữ lại, dict id_bị_loại -> id_được_giữ).
    """
    from pymatgen.analysis.structure_matcher import StructureMatcher
    matcher = StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol)
    keep, dropped = [], {}
    order = df.sort_values(["energy_above_hull_eV_per_atom", "material_id"], na_position="last")
    groups = {}
    for mid in order["material_id"]:
        groups.setdefault(structures[mid].composition.reduced_formula, []).append(mid)
    for ids in groups.values():
        reps = []
        for mid in ids:
            match = next((r for r in reps if matcher.fit(structures[mid], structures[r])), None)
            if match is None:
                reps.append(mid)
                keep.append(mid)
            else:
                dropped[mid] = match
    return keep, dropped


def stratified_select(df, n_max, seed=0, strata_cols=("is_metal", "n_elements_capped")):
    """Chọn tối đa n_max mẫu, giữ tỉ lệ các nhóm (kim loại/không kim loại × số nguyên tố)."""
    if len(df) <= n_max:
        return df.sort_values("material_id").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    parts = []
    for _, g in df.groupby(list(strata_cols)):
        k = max(1, int(round(n_max * len(g) / len(df))))
        parts.append(g.iloc[rng.permutation(len(g))[:k]])
    out = pd.concat(parts)
    if len(out) > n_max:
        out = out.iloc[rng.permutation(len(out))[:n_max]]
    return out.sort_values("material_id").reset_index(drop=True)


def group_split(df, group_col="reduced_formula", fractions=(0.8, 0.1, 0.1), seed=0):
    """Chia train/val/test theo nhóm để các đa hình cùng công thức không bị tách ra nhiều tập.

    Trả về Series tên split ('train' | 'val' | 'test') cùng index với df.
    """
    assert abs(sum(fractions) - 1.0) < 1e-9
    rng = np.random.default_rng(seed)
    groups = df[group_col].unique()
    groups = groups[rng.permutation(len(groups))]
    sizes = df.groupby(group_col).size()
    n = len(df)
    targets = np.cumsum(fractions) * n
    assign, running, bucket = {}, 0, 0
    names = ["train", "val", "test"]
    for g in groups:
        while bucket < 2 and running >= targets[bucket]:
            bucket += 1
        assign[g] = names[bucket]
        running += sizes[g]
    return df[group_col].map(assign)
