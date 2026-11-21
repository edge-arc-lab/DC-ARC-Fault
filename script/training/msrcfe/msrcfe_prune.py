#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi di ablazione e potatura dell'estrattore MS-RCFE e del classificatore Ridge.

Valuta:
- Contributo dei singoli rami convoluzionali (kernel {3,5,9} x dilatazioni {1,2,4}).
- Baseline di riferimento basate su statistiche semplici della finestra temporale.
- Selezione greedy incrementale (5-fold CV sul training set).
- Potatura (pruning) delle feature per magnitudo |coef| * std.
- Profilazione del costo computazionale (operazioni MAC e parametri).
"""
import argparse
import csv
import io
import json
import pickle
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

KS, DS = (3, 5, 9), (1, 2, 4)
BR = [(k, d) for k in KS for d in DS]          # ordine dei rami = ordine di extract_features
NK = 32
T = 1000


def carica_state(path):
    z = zipfile.ZipFile(path)
    pref = z.namelist()[0].split("/")[0]

    class Marker:
        def __init__(self, name):
            self.name = name

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                def rb(storage, off, size, stride, *a):
                    arr = np.frombuffer(z.read(f"{pref}/data/{storage.name}"), dtype=np.float32)
                    return np.lib.stride_tricks.as_strided(arr[off:], shape=tuple(size), strides=tuple(s * 4 for s in stride)).copy()
                return rb
            if name == "OrderedDict":
                import collections
                return collections.OrderedDict
            if name.endswith("Storage"):
                return lambda *a: None
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            return Marker(pid[2])
    return U(io.BytesIO(z.read(f"{pref}/data.pkl"))).load()


def feats(sd, X, bs=512):
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs].astype(np.float32)
        fl = []
        for ci, (k, d) in enumerate(BR):
            W, b = sd[f"convs.{ci}.weight"], sd[f"convs.{ci}.bias"]
            pad = (k // 2) * d
            xp = np.pad(xb, ((0, 0), (pad, pad)))
            y = np.zeros((len(xb), W.shape[0], xb.shape[1]), dtype=np.float32)
            for j in range(k):
                y += W[None, :, 0, j, None] * xp[:, None, j * d:j * d + xb.shape[1]]
            y = np.maximum(y + b[None, :, None], 0)
            fl += [y.max(-1), y.mean(-1)]
        out.append(np.concatenate(fl, 1))
    return np.concatenate(out)


def idx_ramo(r):
    """colonne (max e media) del ramo r nel vettore da 576"""
    return np.arange(r * 2 * NK, (r + 1) * 2 * NK)


def idx_rami(rami):
    return np.concatenate([idx_ramo(r) for r in rami]) if len(rami) else np.array([], int)


def macs(rami):
    return int(sum(NK * BR[r][0] * T for r in rami))


def params(rami):
    return int(sum(NK * BR[r][0] + NK for r in rami))


def auc(y, s):
    from scipy.stats import rankdata
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def valuta(ridge, Ftr, ytr, Fte, yte):
    from sklearn.linear_model import RidgeClassifier
    clf = RidgeClassifier(alpha=1.0).fit(Ftr, ytr)
    s = clf.decision_function(Fte)
    p = (s > 0).astype(int)
    dr = 100 * np.mean(p[yte == 1])
    fpr = 100 * np.mean(p[yte == 0])
    return dict(DR=dr, FPR=fpr, BA=(dr + 100 - fpr) / 2, AUC=auc(yte, s), ok=bool(dr >= 95 and fpr <= 5)), clf


def cv_gruppi(grp, n_fold=5, seed=0):
    esp = np.unique(grp)
    np.random.default_rng(seed).shuffle(esp)
    f = {e: i % n_fold for i, e in enumerate(esp)}
    return np.array([f[g] for g in grp])


def ba_cv(F, y, fold):
    from sklearn.linear_model import RidgeClassifier
    p = np.zeros(len(y), int)
    for k in range(fold.max() + 1):
        tr, te = fold != k, fold == k
        p[te] = RidgeClassifier(alpha=1.0).fit(F[tr], y[tr]).predict(F[te])
    return (np.mean(p[y == 1]) + np.mean(1 - p[y == 0])) / 2


def soglia_livello(Xtr, ytr, Xte, yte):
    """Baseline onesto: una sola soglia sul livello medio, scelta sul train (massimizza l'accuratezza bilanciata), valutata sul test."""
    L = Xtr.mean(1)
    o = np.argsort(L, kind="stable")
    Ls, ys = L[o], ytr[o]
    na, npre = max(1, int((ys == 1).sum())), max(1, int((ys == 0).sum()))
    tpr = np.concatenate(([0], np.cumsum(ys == 1))) / na
    tnr = 1 - np.concatenate(([0], np.cumsum(ys == 0))) / npre
    i = int(np.argmax((tpr + tnr) / 2))
    t = 0.95 if i == 0 else (Ls[i - 1] + (Ls[i] if i < len(Ls) else Ls[-1] + 1e-6)) / 2
    s = -(Xte.mean(1) - t)                      # > 0 se il livello e' sotto soglia
    p = (s > 0).astype(int)
    dr, fpr = 100 * np.mean(p[yte == 1]), 100 * np.mean(p[yte == 0])
    return dict(DR=dr, FPR=fpr, BA=(dr + 100 - fpr) / 2, AUC=auc(yte, s), ok=bool(dr >= 95 and fpr <= 5)), t


def semplici(X):
    m = X.mean(1)
    sd = X.std(1)
    return dict(
        livello=m[:, None],
        livello_std=np.c_[m, sd],
        livello_std_minmax=np.c_[m, sd, X.min(1), X.max(1)],
        livello_std_minmax_dif=np.c_[m, sd, X.min(1), X.max(1), np.abs(np.diff(X, axis=1)).mean(1)],
    )


def selftest():
    rng = np.random.default_rng(0)
    ok = idx_ramo(0)[0] == 0 and idx_ramo(8)[-1] == 575 and len(idx_rami([0, 3])) == 128
    ok &= macs([0]) == 32 * 3 * 1000 and params([8]) == 32 * 9 + 32
    y = np.r_[np.ones(100), np.zeros(100)].astype(int)
    ok &= abs(auc(y, y + rng.normal(0, .1, 200)) - 1) < 0.01
    fold = cv_gruppi(np.array([f"E{i % 20}" for i in range(200)]))
    F = np.c_[y + rng.normal(0, .2, 200), rng.normal(0, 1, 200)]
    ok &= ba_cv(F, y, fold) > 0.9
    X = np.tile(np.linspace(0.9, 1, 1000), (5, 1))
    ok &= semplici(X)["livello_std"].shape == (5, 2)
    Xa = np.vstack([np.full((50, 10), 1.0), np.full((50, 10), 0.8)]); ya = np.r_[np.zeros(50), np.ones(50)].astype(int)
    rr, tt = soglia_livello(Xa, ya, Xa, ya)
    ok &= rr["DR"] == 100 and rr["FPR"] == 0 and 0.8 < tt < 1.0
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--meta-train", default=None)
    ap.add_argument("--extractor", default=None)
    ap.add_argument("--out", default="ris_msrcfe_prune")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    repo = Path(a.repo) if a.repo else Path.cwd()
    tr_p = Path(a.train) if a.train else repo / "dataset/dataset_new/arc_dataset_train.npz"
    te_p = Path(a.test) if a.test else repo / "dataset/dataset_new/arc_dataset_test.npz"
    mt_p = Path(a.meta_train) if a.meta_train else repo / "dataset/dataset_new/arc_dataset_meta_train.csv"
    ex_p = Path(a.extractor) if a.extractor else repo / "script/training/msrcfe/results/msrcfe_extractor.pt"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dtr, dte = np.load(tr_p), np.load(te_p)
    Xtr, ytr, Xte, yte = dtr["X"], dtr["y"].astype(int), dte["X"], dte["y"].astype(int)
    with open(mt_p, newline="", encoding="utf-8") as f:
        grp = np.array([r["exp_key"] for r in csv.DictReader(f)])
    if len(grp) != len(ytr):
        sys.exit("meta-train non allineato al train")
    sd = carica_state(ex_p)
    cache = out / "feat.npz"
    if cache.exists() and np.load(cache)["tr"].shape[0] == len(Xtr) and np.load(cache)["te"].shape[0] == len(Xte):
        Ftr, Fte = np.load(cache)["tr"], np.load(cache)["te"]
    else:
        t0 = time.time()
        Ftr, Fte = feats(sd, Xtr), feats(sd, Xte)
        np.savez(cache, tr=Ftr, te=Fte)
        print(f"feature calcolate in {time.time() - t0:.0f} s")
    fold = cv_gruppi(grp)
    righe = []

    def r(nome, cfg, res, n_feat, rami=None):
        righe.append(dict(analisi=cfg, config=nome, n_feat=n_feat, rami=("" if rami is None else len(rami)),
                          MAC=("" if rami is None else macs(rami)), param=("" if rami is None else params(rami)), **res))
        print(f"{cfg:<3}{nome:<46}{n_feat:>5}{(str(len(rami)) if rami is not None else '-'):>6}"
              f"{res['DR']:>8.2f}{res['FPR']:>8.2f}{res['BA']:>8.2f}{res['AUC']:>9.4f}{'  si' if res['ok'] else '  NO':>5}")

    print(f"{'':<3}{'configurazione':<46}{'feat':>5}{'rami':>6}{'DR%':>8}{'FPR%':>8}{'BA%':>8}{'AUC':>9}{'ok':>5}")
    tutti = list(range(9))
    res, clf_full = valuta(None, Ftr, ytr, Fte, yte)
    r("tutte le feature (riferimento)", "A", res, 576, tutti)
    # nota: le baseline B sotto usano il Ridge con soglia 0 (non tarata): per il livello serve la soglia tarata (riga successiva)
    for nome, G in semplici(Xtr).items():
        Gt = semplici(Xte)[nome]
        res, _ = valuta(None, G, ytr, Gt, yte)
        r(f"baseline: {nome}", "B", res, G.shape[1])
    res, t_liv = soglia_livello(Xtr, ytr, Xte, yte)
    r(f"baseline: UNA soglia sul livello (soglia {t_liv:.3f}, scelta sul train)", "B", res, 1)
    for ri in tutti:
        res, _ = valuta(None, Ftr[:, idx_ramo(ri)], ytr, Fte[:, idx_ramo(ri)], yte)
        r(f"solo ramo k={BR[ri][0]} d={BR[ri][1]}", "C", res, 64, [ri])
    for ri in tutti:
        resto = [x for x in tutti if x != ri]
        res, _ = valuta(None, Ftr[:, idx_rami(resto)], ytr, Fte[:, idx_rami(resto)], yte)
        r(f"senza ramo k={BR[ri][0]} d={BR[ri][1]}", "C", res, 512, resto)
    for tipo, vals, pos in (("kernel", KS, 0), ("dilatazione", DS, 1)):
        for v in vals:
            sel = [i for i, b in enumerate(BR) if b[pos] == v]
            res, _ = valuta(None, Ftr[:, idx_rami(sel)], ytr, Fte[:, idx_rami(sel)], yte)
            r(f"solo {tipo} = {v}", "C", res, 64 * len(sel), sel)
    # D greedy
    scelti, rest = [], list(tutti)
    for step in range(9):
        best = max(rest, key=lambda c: ba_cv(Ftr[:, idx_rami(scelti + [c])], ytr, fold))
        scelti.append(best)
        rest.remove(best)
        res, _ = valuta(None, Ftr[:, idx_rami(scelti)], ytr, Fte[:, idx_rami(scelti)], yte)
        r("greedy: " + ",".join(f"k{BR[x][0]}d{BR[x][1]}" for x in scelti), "D", res, 64 * len(scelti), list(scelti))
    # E feature
    imp = np.abs(clf_full.coef_.ravel()) * Ftr.std(0)
    ordine = np.argsort(-imp)
    for k in (2, 4, 8, 16, 32, 64, 128, 256):
        c = np.sort(ordine[:k])
        res, _ = valuta(None, Ftr[:, c], ytr, Fte[:, c], yte)
        rami_usati = sorted({int(x) // 64 for x in c})
        r(f"top-{k} feature per |coef|*std", "E", res, k, rami_usati)
        if k == 4:
            print("     le 4 feature: " + ", ".join(f"ramo k{BR[int(x) // 64][0]}d{BR[int(x) // 64][1]} {'max' if (int(x) % 64) < NK else 'media'} kernel {int(x) % NK}" for x in c))
    mx = np.concatenate([idx_ramo(i)[:NK] for i in tutti])
    me = np.concatenate([idx_ramo(i)[NK:] for i in tutti])
    for nome, c in (("solo feature 'max'", mx), ("solo feature 'media'", me)):
        res, _ = valuta(None, Ftr[:, c], ytr, Fte[:, c], yte)
        r(nome, "E", res, len(c), tutti)
    with open(out / "ablazione.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader()
        w.writerows(righe)
    print(f"\nScritto {out.resolve() / 'ablazione.csv'}")
    print("Limiti: un solo estrattore (kernel casuali di un singolo seed); Ridge alpha=1 non ottimizzato; test set con livello molto separabile "
          "(vedi parita_livello_v1): le differenze tra configurazioni sono piccole e vanno lette con prudenza.")


if __name__ == "__main__":
    main()
