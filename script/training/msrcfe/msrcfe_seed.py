#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi di stabilita' statistica delle potature di MS-RCFE al variare del seed casuale.

Genera N estrattori casuali (stessa distribuzione di inizializzazione Conv1d)
e ne valuta le metriche su test set per:
- Modello completo (576 feature)
- Ramo ottimale selezionato tramite 5-fold CV sul training set
- Ramo fisso k=3, d=4
- Gruppo di 3 rami con kernel 3
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[2] if (QUI.parents[2] / "dataset").is_dir() else QUI.parent
for p in [str(QUI), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import msrcfe_prune as pr


def estrattore_casuale(pr, seed):
    """Stessa distribuzione dell'init di default di nn.Conv1d(1, 32, k): U(-b, b), b = 1/sqrt(k)."""
    rng = np.random.default_rng(seed)
    sd = {}
    for ci, (k, d) in enumerate(pr.BR):
        b = 1.0 / np.sqrt(k)
        sd[f"convs.{ci}.weight"] = rng.uniform(-b, b, (pr.NK, 1, k)).astype(np.float32)
        sd[f"convs.{ci}.bias"] = rng.uniform(-b, b, (pr.NK,)).astype(np.float32)
    return sd


def selftest():
    sd = estrattore_casuale(pr, 0)
    ok = sd["convs.0.weight"].shape == (32, 1, 3) and abs(float(sd["convs.8.weight"].max()) - 1 / 3) < 0.01
    X = np.random.default_rng(0).normal(1, 0.01, (4, 1000)).astype(np.float32)
    F = pr.feats(sd, X)
    ok &= F.shape == (4, 576) and np.isfinite(F).all()
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--out", default="ris_msrcfe_seed")
    ap.add_argument("--n-seed", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=9000)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    repo = Path(a.repo) if a.repo else REPO
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dtr = np.load(repo / "dataset/dataset_new/arc_dataset_train.npz")
    dte = np.load(repo / "dataset/dataset_new/arc_dataset_test.npz")
    Xtr, ytr, Xte, yte = dtr["X"], dtr["y"].astype(int), dte["X"], dte["y"].astype(int)
    with open(repo / "dataset/dataset_new/arc_dataset_meta_train.csv", newline="", encoding="utf-8") as f:
        grp = np.array([r["exp_key"] for r in csv.DictReader(f)])
    sub = np.sort(np.random.default_rng(0).choice(len(ytr), min(a.n_train, len(ytr)), replace=False))
    Xs, ys, gs = Xtr[sub], ytr[sub], grp[sub]
    fold = pr.cv_gruppi(gs)
    res_liv, t_liv = pr.soglia_livello(Xtr, ytr, Xte, yte)
    print(f"baseline: una soglia sul livello ({t_liv:.3f}): DR {res_liv['DR']:.2f}  FPR {res_liv['FPR']:.2f}  BA {res_liv['BA']:.2f}\n")
    cfgs = ["tutte (576)", "ramo scelto su train (64)", "ramo k3d4 (64)", "3 rami kernel=3 (192)"]
    acc = {c: [] for c in cfgs}
    scelti = []
    print(f"{'seed':>4}  {'ramo scelto':<12}" + "".join(f"{c:>30}" for c in cfgs))
    for seed in range(a.n_seed):
        t0 = time.time()
        sd = estrattore_casuale(pr, 1000 + seed)
        Ftr, Fte = pr.feats(sd, Xs), pr.feats(sd, Xte)
        bas = [pr.ba_cv(Ftr[:, pr.idx_ramo(r)], ys, fold) for r in range(9)]
        rb = int(np.argmax(bas))
        scelti.append(f"k{pr.BR[rb][0]}d{pr.BR[rb][1]}")
        sets = {cfgs[0]: list(range(9)), cfgs[1]: [rb], cfgs[2]: [pr.BR.index((3, 4))],
                cfgs[3]: [i for i, b in enumerate(pr.BR) if b[0] == 3]}
        riga = []
        for c in cfgs:
            idx = pr.idx_rami(sets[c])
            r_, _ = pr.valuta(None, Ftr[:, idx], ys, Fte[:, idx], yte)
            acc[c].append(r_)
            riga.append(f"{r_['DR']:.1f}/{r_['FPR']:.2f}/{r_['BA']:.2f}")
        print(f"{seed:>4}  {scelti[-1]:<12}" + "".join(f"{x:>30}" for x in riga) + f"   ({time.time() - t0:.0f} s)")
    print("\nmedia +- dev.std su", a.n_seed, "seed   (DR % / FPR % / BA %)")
    righe = []
    for c in cfgs:
        m = {k: (np.mean([x[k] for x in acc[c]]), np.std([x[k] for x in acc[c]])) for k in ("DR", "FPR", "BA")}
        print(f"  {c:<30} DR {m['DR'][0]:.2f} +- {m['DR'][1]:.2f}   FPR {m['FPR'][0]:.2f} +- {m['FPR'][1]:.2f}   BA {m['BA'][0]:.2f} +- {m['BA'][1]:.2f}")
        righe.append([c] + [f"{m[k][i]:.3f}" for k in ("DR", "FPR", "BA") for i in (0, 1)])
    print(f"  {'una soglia sul livello':<30} DR {res_liv['DR']:.2f}   FPR {res_liv['FPR']:.2f}   BA {res_liv['BA']:.2f}")
    with open(out / "seed_robustezza.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "DR_media", "DR_sd", "FPR_media", "FPR_sd", "BA_media", "BA_sd"])
        w.writerows(righe)
    print(f"\nrami scelti per seed: {scelti}\nScritto {out.resolve() / 'seed_robustezza.csv'}")
    print("Limiti: Ridge su sottocampione del train; N seed piccolo; una sola divisione train/test; test set molto separabile per livello.")


if __name__ == "__main__":
    main()
