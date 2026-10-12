"""Tiện ích tải dữ liệu từ Materials Project (dùng trong 01_download_mp.ipynb)."""
import getpass
import json
import os

import pandas as pd

# Các khóa cài đặt tính toán giữ lại trong manifest; bản đầy đủ lưu riêng ở mp_settings.jsonl
COMPACT_KEYS = ("incar", "kpoints", "potcar", "potcar_spec", "potcar_type", "is_hubbard", "hubbards")


def get_api_key(secret_name="MP_API_KEY"):
    """Lấy API key từ Colab Secrets, rồi biến môi trường, cuối cùng mới hỏi nhập ẩn.

    `secret_name` là TÊN secret (ví dụ MP_API_KEY), không phải giá trị key.
    """
    key = None
    try:
        from google.colab import userdata
        key = userdata.get(secret_name)
    except Exception:
        key = None
    if not key:
        key = os.environ.get(secret_name)
    if not key:
        key = getpass.getpass(f"Nhập {secret_name} (ẩn, không được lưu vào file): ")
    return key


def get_value(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def to_plain(obj):
    """Chuyển object của API thành kiểu có thể ghi vào JSON/CSV."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if hasattr(obj, "model_dump"):
        return to_plain(obj.model_dump(mode="json"))
    if hasattr(obj, "as_dict"):
        return to_plain(obj.as_dict())
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return str(obj)


def batches(items, size=100):
    for start in range(0, len(items), size):
        yield items[start:start + size]


def compact_settings(settings):
    """Giữ các khóa quan trọng của task input để manifest gọn."""
    if not isinstance(settings, dict):
        return {}
    return {k: settings[k] for k in COMPACT_KEYS if k in settings}


def download_mp(api_key, n_materials=1000, num_sites=(1, 20), stable_only=True,
                e_hull_max=None, include_gnome=False):
    """Tải cấu trúc + metadata. Trả về (rows, database_version).

    rows: list[dict], mỗi dict có thêm `structure_cif` và `settings_full` (bản đầy đủ).
    """
    from mp_api.client import MPRester

    query = dict(num_sites=num_sites, include_gnome=include_gnome, num_chunks=1,
                 chunk_size=n_materials, all_fields=False,
                 fields=["material_id", "formula_pretty", "structure", "origins",
                         "band_gap", "energy_above_hull"])
    if stable_only:
        query["is_stable"] = True
    elif e_hull_max is not None:
        query["energy_above_hull"] = (0, e_hull_max)

    with MPRester(api_key) as mpr:
        database_version = getattr(mpr, "db_version", None) or mpr.get_database_version()
        docs = mpr.materials.summary.search(**query)[:n_materials]
        material_ids = [str(d.material_id) for d in docs]

        # run_type lấy từ entry có task_id trùng với task tạo cấu trúc
        run_type_by_task = {}
        for id_batch in batches(material_ids):
            for thermo_doc in mpr.materials.thermo.search(material_ids=id_batch,
                                                          fields=["material_id", "entries"]):
                for entry in (get_value(thermo_doc, "entries", {}) or {}).values():
                    data = to_plain(get_value(entry, "data", {})) or {}
                    params = to_plain(get_value(entry, "parameters", {})) or {}
                    if data.get("task_id") and params.get("run_type"):
                        run_type_by_task[str(data["task_id"])] = str(params["run_type"])

        # task tạo cấu trúc của từng vật liệu
        task_of = {}
        for doc in docs:
            origins = to_plain(get_value(doc, "origins", [])) or []
            origin = next((o for o in origins if str(o.get("name", "")).lower() == "structure"), None)
            if origin and origin.get("task_id"):
                task_of[str(doc.material_id)] = str(origin["task_id"])

        task_by_id = {}
        for task_batch in batches(sorted(set(task_of.values()))):
            for t in mpr.materials.tasks.search(task_ids=task_batch,
                                                fields=["task_id", "task_type", "input", "last_updated"]):
                task_by_id[str(t.task_id)] = t

    return build_rows(docs, task_of, task_by_id, run_type_by_task, str(database_version)), str(database_version)


def build_rows(docs, task_of, task_by_id, run_type_by_task, database_version):
    rows = []
    drop = {"structure", "initial_structure", "poscar", "contcar"}
    for doc in docs:
        mid = str(doc.material_id)
        task_id = task_of.get(mid)
        task = task_by_id.get(task_id) if task_id else None
        inp = (to_plain(get_value(task, "input", {})) if task else {}) or {}
        settings = {k: v for k, v in inp.items() if k.lower() not in drop} if isinstance(inp, dict) else {}
        structure = get_value(doc, "structure")
        rows.append({
            "material_id": mid,
            "formula_pretty": get_value(doc, "formula_pretty"),
            "task_id": task_id,
            "run_type": run_type_by_task.get(task_id, "UNKNOWN") if task_id else "UNKNOWN",
            "task_type": to_plain(get_value(task, "task_type")) if task else None,
            "task_last_updated": to_plain(get_value(task, "last_updated")) if task else None,
            "mp_database_version": database_version,
            "band_gap_eV": get_value(doc, "band_gap"),
            "energy_above_hull_eV_per_atom": get_value(doc, "energy_above_hull"),
            "n_sites": len(structure) if structure is not None else None,
            "settings_compact_json": json.dumps(to_plain(compact_settings(settings)),
                                                ensure_ascii=False, separators=(",", ":")),
            "structure_cif": structure.to(fmt="cif") if structure is not None else None,
            "settings_full": to_plain(settings),
        })
    return rows


def save_raw(rows, raw_dir):
    """Ghi raw/mp_manifest.csv, raw/cif/<id>.cif và raw/mp_settings.jsonl."""
    cif_dir = os.path.join(raw_dir, "cif")
    os.makedirs(cif_dir, exist_ok=True)
    manifest = []
    with open(os.path.join(raw_dir, "mp_settings.jsonl"), "w") as fs:
        for r in rows:
            mid = r["material_id"]
            if r["structure_cif"]:
                with open(os.path.join(cif_dir, f"{mid}.cif"), "w") as f:
                    f.write(r["structure_cif"])
            fs.write(json.dumps({"material_id": mid, "task_id": r["task_id"],
                                 "settings": r["settings_full"]}, ensure_ascii=False) + "\n")
            row = {k: v for k, v in r.items() if k not in ("structure_cif", "settings_full")}
            row["cif_path"] = f"cif/{mid}.cif" if r["structure_cif"] else None
            manifest.append(row)
    df = pd.DataFrame(manifest)
    df.to_csv(os.path.join(raw_dir, "mp_manifest.csv"), index=False)
    return df
