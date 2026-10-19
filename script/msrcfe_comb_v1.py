#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
msrcfe_comb_v1.py - riaddestra il Ridge di MS-RCFE su TUTTE le 511 combinazioni dei 9 rami convoluzionali
=========================================================================================================

Per ogni sottoinsieme non vuoto dei 9 rami (kernel {3,5,9} x dilatazione {1,2,4}; 2^9-1 = 511 combinazioni) rifa' il
RidgeClassifier (alpha=1, come train_msrcfe.py) sulle sole feature dei rami tenuti (max+media di 32 kernel per ramo = 64 feature a ramo)
e riporta, per ciascuna:
  - DR, FPR, accuratezza bilanciata (BA), AUC sul TEST e se rispetta DR >= 95% e FPR <= 5%;
  - BA in validazione 5-fold per ESPERIMENTO sul solo TRAIN (e' il criterio con cui si sceglie: il test non si usa per scegliere);
  - costo: MAC e parametri dell'estrattore, frazione di MAC rispetto al modello completo e latenza STIMATA per proporzione
    (--lat-fp32 / --lat-int8 della scheda: e' solo un'indicazione, la latenza vera va MISURATA sulla scheda).
Il Ridge e' risolto in forma chiusa dalla matrice di Gram completa (sottoblocchi): le 511 combinazioni x 5 fold costano pochi secondi,
e il risultato e' identico a sklearn (verificato dal selftest).

Estrattore: quello pubblicato (default) oppure N estrattori casuali con seed diversi (--n-seed N, stessa distribuzione dell'init di nn.Conv1d)
per vedere se la combinazione migliore e' la stessa al variare dei kernel casuali. Con piu' seed si scrive anche l'aggregato (media +- dev.std per
combinazione) e la combinazione scelta dalla validazione in ogni seed, con le sue prestazioni sul test.

Uscite in --out
  combinazioni_<tag>.csv      una riga per combinazione (per seed)
  aggregato.csv               (solo con --n-seed > 1) media +- dev.std su seed per combinazione
  scelta_validazione.csv      combinazione scelta sul train (BA CV massima, a parita' meno MAC) e sua resa sul test, per seed
  pareto.csv                  frontiera BA(test) contro MAC (per l'estrattore pubblicato o la media sui seed)

Uso (dalla cartella della repo)
    python -B script/msrcfe_comb_v1.py --repo . --out ..\ris_comb
    python -B script/msrcfe_comb_v1.py --repo . --out ..\ris_comb_seed --n-seed 5
    python -B script/msrcfe_comb_v1.py --repo . --train ..\dataset_V\arc_dataset_train_V.npz --test ..\dataset_V\arc_dataset_test_V.npz --meta-train ..\dataset_V\arc_dataset_meta_train_V.csv --out ..\ris_comb_V
    python -B script/msrcfe_comb_v1.py --selftest
Richiede msrcfe_prune_v1.py e msrcfe_seed_v1.py nella stessa cartella.
Limiti: kernel casuali non addestrati (seed-dipendenti); Ridge alpha=1 non ottimizzato; test molto separabile per livello
(vedi parita_livello_v1): differenze piccole tra combinazioni non sono significative; nessuna misura su scheda.
"""
import argparse
import csv
import importlib.util
import itertools
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent
ALPHA = 1.0


def carica(nome):
    sp = importlib.util.spec_from_file_location(nome, QUI / f"{nome}.py")
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


class Gram:
    """Statistiche per il Ridge con intercetta (bersagli +-1) su un insieme di campioni."""

    def __init__(self, F, t):
        F = F.astype(np.float64)
        self.n = len(F)
        self.S = F.T @ F
        self.s = F.sum(0)
        self.b = F.T @ t
        self.ts = float(t.sum())

    def sottrai(self, o):
        g = object.__new__(Gram)
        g.n, g.S, g.s, g.b, g.ts = self.n - o.n, self.S - o.S, self.s - o.s, self.b - o.b, self.ts - o.ts
        return g

    def risolvi(self, idx, alpha=ALPHA):
        n = self.n
        A = self.S[np.ix_(idx, idx)] - np.outer(self.s[idx], self.s[idx]) / n
        b = self.b[idx] - self.s[idx] * self.ts / n
        w = np.linalg.solve(A + alpha * np.eye(len(idx)), b)
        b0 = self.ts / n - (self.s[idx] / n) @ w
        return w, b0


def tutte_combinazioni(pr):
    nr = len(pr.BR)
    return [c for k in range(1, nr + 1) for c in itertools.combinations(range(nr), k)]


def nome_comb(pr, c):
    return "+".join(f"k{pr.BR[r][0]}d{pr.BR[r][1]}" for r in c)


def metriche(pr, y, s):
    p = s > 0
    dr, fpr = 100 * np.mean(p[y == 1]), 100 * np.mean(p[y == 0])
    return dr, fpr, (dr + 100 - fpr) / 2, pr.auc(y, s)


def analizza(pr, Ftr, ytr, Fte, yte, fold, combs):
    ttr = 2.0 * ytr - 1.0
    G = Gram(Ftr, ttr)
    nf = int(fold.max()) + 1
    Gf, Gtr_f = [], []
    for k in range(nf):
        m = fold == k
        g = Gram(Ftr[m], ttr[m])
        Gf.append(g)
        Gtr_f.append(G.sottrai(g))
    Ftr64 = Ftr.astype(np.float64)
    righe = []
    for c in combs:
        idx = pr.idx_rami(list(c))
        w, b0 = G.risolvi(idx)
        s_te = Fte[:, idx].astype(np.float64) @ w + b0
        dr, fpr, ba, au = metriche(pr, yte, s_te)
        pred = np.zeros(len(ytr), bool)
        for k in range(nf):
            m = fold == k
            wk, bk = Gtr_f[k].risolvi(idx)
            pred[m] = (Ftr64[m][:, idx] @ wk + bk) > 0
        ba_cv = (100 * np.mean(pred[ytr == 1]) + 100 * np.mean(~pred[ytr == 0])) / 2
        righe.append(dict(n_rami=len(c), rami=nome_comb(pr, c), n_feat=len(idx), MAC=pr.macs(list(c)), param=pr.params(list(c)),
                          BA_cv_train=ba_cv, DR=dr, FPR=fpr, BA=ba, AUC=au, ok=int(dr >= 95 and fpr <= 5)))
    return righe


def aggiungi_costi(righe, lat32, lat8, macs_full):
    for r in righe:
        fr = r["MAC"] / macs_full
        r["frac_MAC"] = fr
        r["lat_stim_fp32_ms"] = lat32 * fr
        r["lat_stim_int8_ms"] = lat8 * fr


def scegli(righe):
    """combinazione scelta SOLO dal train: BA CV massima, a parita' (entro 0.05) meno MAC"""
    mx = max(r["BA_cv_train"] for r in righe)
    cand = [r for r in righe if r["BA_cv_train"] >= mx - 0.05]
    return min(cand, key=lambda r: (r["MAC"], -r["BA_cv_train"]))


def pareto(righe, chiave_ba="BA"):
    o = sorted(righe, key=lambda r: (r["MAC"], -r[chiave_ba]))
    out, best = [], -1
    for r in o:
        if r[chiave_ba] > best + 1e-12:
            out.append(r)
            best = r[chiave_ba]
    return out


def scrivi(path, righe, campi=None):
    campi = campi or list(righe[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campi, extrasaction="ignore")
        w.writeheader()
        for r in righe:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})


def selftest():
    pr = carica("msrcfe_prune_v1")
    from sklearn.linear_model import RidgeClassifier
    rng = np.random.default_rng(0)
    n = 600
    y = (rng.random(n) > 0.5).astype(int)
    F = rng.normal(0, 1, (n, 576)) + y[:, None] * rng.normal(0, 0.3, 576)
    t = 2.0 * y - 1
    G = Gram(F, t)
    ok = True
    for c in [(0,), (2, 5), tuple(range(9))]:
        idx = pr.idx_rami(list(c))
        w, b0 = G.risolvi(idx)
        clf = RidgeClassifier(alpha=1.0).fit(F[:, idx], y)
        ok &= np.allclose(F[:, idx] @ w + b0, clf.decision_function(F[:, idx]), atol=1e-6)
    m = np.arange(n) % 5 == 0
    g1 = G.sottrai(Gram(F[m], t[m]))
    idx = pr.idx_rami([1, 4])
    w, b0 = g1.risolvi(idx)
    clf = RidgeClassifier(alpha=1.0).fit(F[~m][:, idx], y[~m])
    ok &= np.allclose(F[:, idx] @ w + b0, clf.decision_function(F[:, idx]), atol=1e-6)
    ok &= len(tutte_combinazioni(pr)) == 511
    ok &= nome_comb(pr, (0, 8)) == "k3d1+k9d4"
    righe = [dict(MAC=3, BA=90.0, BA_cv_train=90.0), dict(MAC=1, BA=80.0, BA_cv_train=80.0),
             dict(MAC=2, BA=85.0, BA_cv_train=99.0), dict(MAC=4, BA=89.0, BA_cv_train=99.0)]
    ok &= [r["MAC"] for r in pareto(righe)] == [1, 2, 3]
    ok &= scegli(righe)["MAC"] == 2
    fold = pr.cv_gruppi(np.array([f"E{i % 25}" for i in range(n)]))
    r = analizza(pr, F[:, :576], y, F, y, fold, [(0,), (1, 2)])
    ok &= len(r) == 2 and r[0]["n_feat"] == 64 and r[1]["n_feat"] == 128
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--test", default=None)
    ap.add_argument("--meta-train", default=None)
    ap.add_argument("--extractor", default=None)
    ap.add_argument("--n-seed", type=int, default=0, help="0 = solo estrattore pubblicato; N>0 = N estrattori casuali (seed 1000..)")
    ap.add_argument("--n-train", type=int, default=0, help="sottocampiona il train a N finestre (0 = tutto)")
    ap.add_argument("--lat-fp32", type=float, default=84.75, help="latenza MS-RCFE FP32 misurata sulla H7S78 (ms), per la stima proporzionale")
    ap.add_argument("--lat-int8", type=float, default=69.77)
    ap.add_argument("--out", default="ris_comb")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    pr, sm = carica("msrcfe_prune_v1"), carica("msrcfe_seed_v1")
    repo = Path(a.repo) if a.repo else Path.cwd()
    tr_p = Path(a.train) if a.train else repo / "dataset/dataset_new/arc_dataset_train.npz"
    te_p = Path(a.test) if a.test else repo / "dataset/dataset_new/arc_dataset_test.npz"
    mt_p = Path(a.meta_train) if a.meta_train else repo / "dataset/dataset_new/arc_dataset_meta_train.csv"
    ex_p = Path(a.extractor) if a.extractor else repo / "script/training/msrcfe/results/msrcfe_extractor.pt"
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    dtr, dte = np.load(tr_p), np.load(te_p)
    Xtr, ytr, Xte, yte = dtr["X"], dtr["y"].astype(int), dte["X"], dte["y"].astype(int)
    with open(mt_p, newline="", encoding="utf-8") as f:
        grp = np.array([r["exp_key"] for r in csv.DictReader(f)])
    if len(grp) != len(ytr):
        sys.exit("meta-train non allineato al train")
    if a.n_train and a.n_train < len(ytr):
        sub = np.sort(np.random.default_rng(0).choice(len(ytr), a.n_train, replace=False))
        Xtr, ytr, grp = Xtr[sub], ytr[sub], grp[sub]
    fold = pr.cv_gruppi(grp)
    combs = tutte_combinazioni(pr)
    macs_full = pr.macs(list(range(9)))
    sorgenti = [("pubblicato", None)] if a.n_seed == 0 else [(f"seed{1000 + i}", 1000 + i) for i in range(a.n_seed)]
    tutte, scelte = {}, []
    for tag, seed in sorgenti:
        t0 = time.time()
        if seed is None:
            sd = pr.carica_state(ex_p)
            cache = out / "feat_pubblicato.npz"
            if cache.exists() and np.load(cache)["tr"].shape[0] == len(Xtr) and np.load(cache)["te"].shape[0] == len(Xte):
                z = np.load(cache)
                Ftr, Fte = z["tr"], z["te"]
            else:
                Ftr, Fte = pr.feats(sd, Xtr), pr.feats(sd, Xte)
                np.savez(cache, tr=Ftr, te=Fte)
        else:
            sd = sm.estrattore_casuale(pr, seed)
            Ftr, Fte = pr.feats(sd, Xtr), pr.feats(sd, Xte)
        righe = analizza(pr, Ftr, ytr, Fte, yte, fold, combs)
        aggiungi_costi(righe, a.lat_fp32, a.lat_int8, macs_full)
        tutte[tag] = righe
        scrivi(out / f"combinazioni_{tag}.csv", righe)
        sc = scegli(righe)
        completo = [r for r in righe if r["n_rami"] == 9][0]
        scelte.append(dict(estrattore=tag, scelta_su_train=sc["rami"], n_rami=sc["n_rami"], MAC=sc["MAC"], frac_MAC=sc["frac_MAC"],
                           BA_cv_train=sc["BA_cv_train"], DR_test=sc["DR"], FPR_test=sc["FPR"], BA_test=sc["BA"], ok=sc["ok"],
                           BA_test_completo=completo["BA"], DR_test_completo=completo["DR"], FPR_test_completo=completo["FPR"]))
        n_ok = sum(r["ok"] for r in righe)
        mig = max(righe, key=lambda r: r["BA"])
        print(f"[{tag}] {len(righe)} combinazioni in {time.time() - t0:.0f} s | rispettano DR>=95/FPR<=5: {n_ok} | "
              f"completo BA {completo['BA']:.2f} | migliore sul test (solo informativo) {mig['rami']} BA {mig['BA']:.2f} | "
              f"scelta sul train {sc['rami']} -> DR {sc['DR']:.2f} FPR {sc['FPR']:.2f} BA {sc['BA']:.2f}, MAC {100 * sc['frac_MAC']:.0f}%")
    scrivi(out / "scelta_validazione.csv", scelte)
    # aggregato su seed
    rif = tutte[sorgenti[0][0]]
    if len(sorgenti) > 1:
        agg = []
        for i, r0 in enumerate(rif):
            v = [tutte[t][i] for t, _ in sorgenti]
            d = dict(n_rami=r0["n_rami"], rami=r0["rami"], MAC=r0["MAC"], frac_MAC=r0["frac_MAC"],
                     lat_stim_fp32_ms=r0["lat_stim_fp32_ms"], lat_stim_int8_ms=r0["lat_stim_int8_ms"])
            for k in ("DR", "FPR", "BA", "BA_cv_train"):
                d[k + "_media"] = float(np.mean([x[k] for x in v]))
                d[k + "_sd"] = float(np.std([x[k] for x in v]))
            d["n_seed_ok"] = int(sum(x["ok"] for x in v))
            agg.append(d)
        scrivi(out / "aggregato.csv", agg)
        par = pareto([dict(r, BA=r["BA_media"]) for r in agg])
        print(f"\nAGGREGATO su {len(sorgenti)} seed: frontiera BA(test) contro MAC (combinazioni con meno MAC e BA piu' alta)")
        for r in par:
            print(f"  {r['rami']:<40}{100 * r['frac_MAC']:>6.0f}% MAC  BA {r['BA_media']:.2f} +- {r['BA_sd']:.2f}  DR {r['DR_media']:.2f}  FPR {r['FPR_media']:.2f}  "
                  f"ok in {r['n_seed_ok']}/{len(sorgenti)} seed")
        scrivi(out / "pareto.csv", par, ["n_rami", "rami", "MAC", "frac_MAC", "lat_stim_fp32_ms", "lat_stim_int8_ms",
                                         "BA_media", "BA_sd", "DR_media", "FPR_media", "n_seed_ok"])
    else:
        par = pareto(rif)
        print("\nFRONTIERA BA(test) contro MAC (estrattore pubblicato)")
        for r in par:
            print(f"  {r['rami']:<40}{100 * r['frac_MAC']:>6.0f}% MAC  lat.stim. {r['lat_stim_fp32_ms']:.0f}/{r['lat_stim_int8_ms']:.0f} ms  "
                  f"DR {r['DR']:.2f}  FPR {r['FPR']:.2f}  BA {r['BA']:.2f}  (BA CV train {r['BA_cv_train']:.2f})")
        scrivi(out / "pareto.csv", par)
    # quali combinazioni rientrano nel passo da 50 ms (stima proporzionale)
    ok50 = [r for r in rif if r["lat_stim_int8_ms"] < 50 and r["ok"]]
    print(f"\nCombinazioni con latenza INT8 STIMATA < 50 ms e DR>=95/FPR<=5 ({sorgenti[0][0]}): {len(ok50)}. La stima e' proporzionale ai MAC: "
          f"costi fissi (feature, Ridge, I/O) non inclusi, va misurata sulla scheda.")
    print(f"\nScritto in {out}")
    print("Limiti: kernel casuali non addestrati; Ridge alpha=1; test molto separabile per livello; differenze piccole non significative.")


if __name__ == "__main__":
    main()
