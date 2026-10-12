"""Mô hình ML-XC bằng JAX (07_train_ml_xc.ipynb, 08_evaluate.ipynb).

Mô hình học hiệu chỉnh XC phụ thuộc cục bộ vào (ρ, s):
    ΔE_θ = ∫ ρ(r) f_θ(ρ(r), s(r)) dr          [Ha, trên một ô đơn vị]
và so với nhãn  Δ = E_ref[ρ_PBE] − E_PBE[ρ_PBE]. Lớp cuối khởi tạo bằng 0 nên mô hình ban đầu chính là PBE.
Mô hình chỉ thấy ρ và |∇ρ|, không thấy τ hay các đại lượng khác mà functional tham chiếu có thể dùng.
"""
import json

import numpy as np

HA2EV = 27.211386245988


def _jax():
    import jax
    import jax.numpy as jnp
    import optax
    return jax, jnp, optax


# ---------------------------------------------------------------- mô hình
def init_params(seed=0, hidden=(32, 32), n_in=2):
    jax, jnp, _ = _jax()
    sizes = [n_in, *hidden, 1]
    keys = jax.random.split(jax.random.PRNGKey(seed), len(sizes) - 1)
    params = []
    for i, (k, a, b) in enumerate(zip(keys, sizes[:-1], sizes[1:])):
        if i == len(sizes) - 2:                     # lớp cuối = 0 -> ΔE ban đầu = 0
            params.append({"W": jnp.zeros((a, b)), "b": jnp.zeros((b,))})
        else:
            params.append({"W": jax.random.normal(k, (a, b)) * jnp.sqrt(2.0 / (a + b)), "b": jnp.zeros((b,))})
    return params


def point_input(rho, s):
    _, jnp, _ = _jax()
    return jnp.stack([(jnp.log10(rho) + 4.0) / 3.0, jnp.arcsinh(s)], axis=-1)


def mlp(params, x):
    _, jnp, _ = _jax()
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["W"] + layer["b"])
    return (x @ params[-1]["W"] + params[-1]["b"])[..., 0]


def delta_e_per_atom_eV(params, rho, s, w, n_atoms):
    """ΔE dự đoán (eV/nguyên tử) cho một mẫu; rho, s phẳng; w = thể tích điểm lưới (Bohr^3)."""
    _, jnp, _ = _jax()
    f = mlp(params, point_input(rho, s))
    return HA2EV * w * jnp.sum(rho * f) / n_atoms


# ---------------------------------------------------------------- dữ liệu
def make_dataset(samples, features_fn):
    """samples: list dict từ data_utils.load_sample; features_fn(rho, lattice_A) -> dict rho, s, w."""
    _, jnp, _ = _jax()
    ds = []
    for smp in samples:
        ft = features_fn(smp["rho"], smp["lattice_A"])
        n = smp["n_atoms"]
        ds.append({"id": smp["material_id"], "rho": jnp.asarray(ft["rho"]), "s": jnp.asarray(ft["s"]),
                   "w": float(ft["w"]), "n": float(n),
                   "target_eV": float(smp["delta_ref_pbe_Ha"]) * HA2EV / n,
                   "pbe_err_eV": float(smp["delta_ref_pbe_Ha"]) * HA2EV / n,
                   "lda_eV": (float(smp["e_lda_Ha"]) - float(smp["e_pbe_Ha"])) * HA2EV / n})
    return ds


# ---------------------------------------------------------------- huấn luyện
def predict(params, ds):
    jax, _, _ = _jax()
    f = jax.jit(delta_e_per_atom_eV)
    return np.array([float(f(params, d["rho"], d["s"], d["w"], d["n"])) for d in ds])


def metrics(pred, target):
    err = np.asarray(pred) - np.asarray(target)
    return {"mae_meV": float(np.mean(np.abs(err)) * 1e3), "rmse_meV": float(np.sqrt(np.mean(err ** 2)) * 1e3),
            "max_meV": float(np.max(np.abs(err)) * 1e3), "n": int(len(err))}


def train(train_ds, val_ds, hidden=(32, 32), lr=1e-3, epochs=200, batch_size=8, patience=30,
          seed=0, verbose=True):
    """Fit trên train_ds; chọn tham số tốt nhất theo RMSE trên val_ds (không dùng test)."""
    jax, jnp, optax = _jax()
    params = init_params(seed, hidden)
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adam(lr))   # cắt gradient để huấn luyện ổn định
    state = opt.init(params)

    def loss_fn(p, rho, s, w, n, tgt):
        return (delta_e_per_atom_eV(p, rho, s, w, n) - tgt) ** 2

    vg = jax.jit(jax.value_and_grad(loss_fn))
    rng = np.random.default_rng(seed)
    best, best_params, bad, history = np.inf, params, 0, []
    for ep in range(epochs):
        order = rng.permutation(len(train_ds))
        tot = 0.0
        for start in range(0, len(order), batch_size):
            grads = None
            idx = order[start:start + batch_size]
            for i in idx:
                d = train_ds[i]
                l, g = vg(params, d["rho"], d["s"], d["w"], d["n"], d["target_eV"])
                tot += float(l)
                g = jax.tree_util.tree_map(lambda x: x / len(idx), g)
                grads = g if grads is None else jax.tree_util.tree_map(jnp.add, grads, g)
            updates, state = opt.update(grads, state, params)
            params = optax.apply_updates(params, updates)
        tr_rmse = float(np.sqrt(tot / len(train_ds)))
        va = metrics(predict(params, val_ds), [d["target_eV"] for d in val_ds])
        history.append({"epoch": ep, "train_rmse_eV": tr_rmse, "val_rmse_meV": va["rmse_meV"]})
        if va["rmse_meV"] < best - 1e-9:
            best, best_params, bad = va["rmse_meV"], params, 0
        else:
            bad += 1
        if verbose and (ep % 10 == 0 or bad == 0):
            print(f"epoch {ep:4d}  train RMSE {tr_rmse * 1e3:8.2f} meV/at  val RMSE {va['rmse_meV']:8.2f} meV/at")
        if bad >= patience:
            break
    return best_params, {"best_val_rmse_meV": best, "history": history, "hidden": list(hidden), "lr": lr}


# ---------------------------------------------------------------- lưu / nạp
def save_model(path, params, config):
    flat = {}
    for i, layer in enumerate(params):
        flat[f"W{i}"] = np.asarray(layer["W"])
        flat[f"b{i}"] = np.asarray(layer["b"])
    flat["config_json"] = np.array(json.dumps(config, default=float))
    np.savez(path, **flat)


def load_model(path):
    _, jnp, _ = _jax()
    d = np.load(path, allow_pickle=False)
    n = len([k for k in d.files if k.startswith("W")])
    params = [{"W": jnp.asarray(d[f"W{i}"]), "b": jnp.asarray(d[f"b{i}"])} for i in range(n)]
    return params, json.loads(str(d["config_json"]))
