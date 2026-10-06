"""Stage 5: train a small CNN on log RD maps; paper split vs grouped split, filtered vs unfiltered.

Experiments (each over SEEDS):
  task=group (drone/bird/human/reflector): train set rd_clean or rd_all  x  split paper or grouped
  task=drone (D1-D6):                      train set rd_clean            x  split paper or grouped
Every model is evaluated on the test split of rd_clean (same rows for both training sets)
and on the test split of rd_all. Model selection: epoch with best val macro-F1.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn

from rdp import config as C
from rdp import hw
from rdp.dataset import RadarRDDataset
from rdp.models import RDClassifier


def _predict(model, x, batch=4096):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch):
            out.append(model(x[i : i + batch]).argmax(1).cpu())
    return torch.cat(out).numpy()


def train_one(train_root, task, split_kind, seed, eval_roots, epochs=12, batch=256, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr = RadarRDDataset(train_root, split_kind, 1, task)
    va = RadarRDDataset(train_root, split_kind, 2, task)
    n_cls = len(tr.classes)
    xtr, ytr = tr.tensors(device)
    xva, yva = va.tensors(device)
    freq = np.bincount(tr.y, minlength=n_cls).astype(np.float64)
    w = torch.tensor(np.where(freq > 0, 1 / np.sqrt(np.maximum(freq, 1)), 0), dtype=torch.float32, device=device)
    w = w / w[w > 0].mean()
    model = RDClassifier(n_cls).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    steps = epochs * ((len(xtr) + batch - 1) // batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps)
    lossf = nn.CrossEntropyLoss(weight=w)
    g = torch.Generator().manual_seed(seed)
    best = (-1.0, None, -1)
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(xtr), generator=g).to(device)
        for i in range(0, len(xtr), batch):
            idx = perm[i : i + batch]
            loss = lossf(model(xtr[idx]), ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        f1 = f1_score(yva.cpu().numpy(), _predict(model, xva), average="macro", labels=np.unique(va.y))
        if f1 > best[0]:
            best = (f1, {k: v.detach().clone() for k, v in model.state_dict().items()}, ep)
    if device == "cuda":
        torch.cuda.synchronize()
    train_s = time.perf_counter() - t0
    model.load_state_dict(best[1])
    res = {
        "train_seconds": round(train_s, 2),
        "best_epoch": best[2],
        "val_macro_f1": round(float(best[0]), 4),
        "n_train": len(tr),
        "n_val": len(va),
        "eval": {},
    }
    for name, root in eval_roots.items():
        te = RadarRDDataset(root, split_kind, 3, task)
        xte, yte = te.tensors(device)
        pred = _predict(model, xte)
        labs = list(range(n_cls))
        present = sorted(set(te.y.tolist()))
        res["eval"][name] = {
            "n_test": len(te),
            "accuracy": round(float(accuracy_score(te.y, pred)), 4),
            "macro_f1": round(float(f1_score(te.y, pred, average="macro", labels=present)), 4),
            "confusion": confusion_matrix(te.y, pred, labels=labs).tolist(),
            "classes": te.classes,
        }
    return res


def run(p=None, seeds=C.SEEDS, epochs: int = 12) -> dict:
    p = p or C.paths()
    clean = p.exports / "rd_clean-v1.0.0"
    allr = p.exports / "rd_all-v1.0.0"
    evals = {"clean_test": clean, "all_test": allr}
    exps = []
    for split_kind in ("paper", "grouped"):
        exps.append(("group", split_kind, "rd_clean", clean))
        exps.append(("group", split_kind, "rd_all", allr))
        exps.append(("drone", split_kind, "rd_clean", clean))
    runs = []
    for task, split_kind, train_name, root in exps:
        for s in seeds:
            r = train_one(root, task, split_kind, s, evals, epochs=epochs)
            r.update(task=task, split=split_kind, train_set=train_name, seed=s)
            runs.append(r)
            print(
                f"[train] {task:5s} {split_kind:7s} {train_name:8s} seed={s} "
                f"clean acc={r['eval']['clean_test']['accuracy']:.3f} f1={r['eval']['clean_test']['macro_f1']:.3f} "
                f"({r['train_seconds']:.1f}s)",
                flush=True,
            )
    summary = {}
    for task, split_kind, train_name, _ in exps:
        rs = [r for r in runs if (r["task"], r["split"], r["train_set"]) == (task, split_kind, train_name)]
        key = f"{task}|{split_kind}|{train_name}"
        summary[key] = {}
        for ev in evals:
            acc = [r["eval"][ev]["accuracy"] for r in rs]
            f1 = [r["eval"][ev]["macro_f1"] for r in rs]
            summary[key][ev] = {
                "accuracy_mean": round(float(np.mean(acc)), 4),
                "accuracy_std": round(float(np.std(acc)), 4),
                "macro_f1_mean": round(float(np.mean(f1)), 4),
                "macro_f1_std": round(float(np.std(f1)), 4),
                "n_seeds": len(rs),
                "n_test": rs[0]["eval"][ev]["n_test"],
            }
        summary[key]["train_seconds_mean"] = round(float(np.mean([r["train_seconds"] for r in rs])), 2)
    res = {
        "protocol": __doc__.strip(),
        "epochs": epochs,
        "seeds": list(seeds),
        "summary": summary,
        "runs": runs,
        "hardware": hw.describe(),
    }
    (p.results / "classifier.json").write_text(json.dumps(res, indent=2))
    return res
