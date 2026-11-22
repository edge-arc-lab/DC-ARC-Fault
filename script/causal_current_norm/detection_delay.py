#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
detection_delay.py
Calcolo del ritardo di rilevamento dell'arco e verifica conformita' UL 1699B.
Simula la decisione a finestra scorrevole sul test set con criterio a K conferme.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
sys.path.insert(0, str(QUI))
import causal_norm as nc  # noqa: E402


def prima_rilevazione(dec, ends, t1_idx, K):
    """Restituisce l'indice della finestra della prima serie di K decisioni positive."""
    n = len(dec)
    for j in range(n - K + 1):
        if ends[j] > t1_idx and dec[j:j + K].all():
            return j + K - 1
    return None


def selftest(p):
    fs, win, step = p.FS_HZ, p.WIN_N, p.STEP_N
    t1 = 20000
    n = 40000
    ends = np.arange(0, n - win + 1, step) + win
    dec = ends > t1 + 3000
    ok = True
    for K in (1, 2, 3):
        j = prima_rilevazione(dec, ends, t1, K)
        ritardo = (ends[j] - t1) / fs
        atteso_fine = ends[np.argmax(dec)] + (K - 1) * step
        ok &= j is not None and ends[j] == atteso_fine
        print(f"K = {K}: ritardo = {ritardo * 1000:.0f} ms")
    dec2 = np.zeros_like(dec)
    dec2[ends <= t1] = True
    ok &= prima_rilevazione(dec2, ends, t1, 1) is None
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_ritardo")
    ap.add_argument("--k", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--latenza-ms", type=float, default=0.0, help="tempo di inferenza [ms]")
    ap.add_argument("--nominale", choices=["statico", "causale"], default="statico")
    ap.add_argument("--W", type=float, default=2.0)
    ap.add_argument("--gate-frac", type=float, default=0.98)
    ap.add_argument("--modelli", nargs="+", default=["inc_fp32", "ms_fp32", "inc_int8", "ms_int8"],
                    choices=["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"])
    ap.add_argument("--downsample-factor", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    p = nc.carica_parametri(repo)
    if args.selftest:
        sys.exit(selftest(p))
    if not args.mat_root:
        sys.exit("serve --mat-root (cartella dei .mat a 10 kHz) oppure --selftest")

    sys.path.insert(0, str(repo / "script"))
    import test_set_analysis as a
    from scipy.io import loadmat

    with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
        files = sorted({m["filename"] for m in csv.DictReader(f)})
    mats = nc.indice_mat(args.mat_root)
    mancanti = [x for x in files if x not in mats]
    if mancanti:
        sys.exit(f"{len(mancanti)} file .mat non trovati in {args.mat_root}, es. {mancanti[:3]}")
    sess = {k: a.apri_sessione(repo / a.MODELLI[k]) for k in args.modelli}
    coef = np.load(repo / a.RIDGE_COEF).astype(np.float64).reshape(-1)
    itc = float(np.load(repo / a.RIDGE_INTERCEPT).reshape(-1)[0])
    fs, win, step = p.FS_HZ, p.WIN_N, p.STEP_N

    def punteggi(k, X):
        return a.score_inception(sess[k], X) if k.startswith("inc") else a.score_msrcfe(sess[k], X, coef, itc)

    righe, senza_arco = [], []
    for fn in files:
        i = loadmat(mats[fn], squeeze_me=False)["CurrentData"].flatten()[::args.downsample_factor].astype(np.float64)
        t1, nom = p.find_t1(i)
        if t1 is None:
            senza_arco.append(fn)
            continue
        starts = np.arange(0, len(i) - win + 1, step)
        ends = starts + win
        if args.nominale == "statico":
            den = np.full(len(starts), nom)
        else:
            by = {s: v for s, v, _ in nc.nominale_causale(i, args.W, "level", p, None, args.gate_frac)}
            den = np.array([by[s] for s in starts])
        X = np.stack([i[s:s + win] / d for s, d in zip(starts, den)]).astype(np.float32)
        for k in args.modelli:
            dec = punteggi(k, X) > 0
            fa_pre = int(((ends <= t1) & dec).sum())
            for K in args.k:
                j = prima_rilevazione(dec, ends, t1, K)
                rit = None if j is None else (ends[j] - t1) / fs * 1000 + args.latenza_ms
                righe.append(dict(file=fn, modello=k, K=K, t1_s=t1 / fs, ritardo_ms=rit, falsi_allarmi_pre=fa_pre))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "ritardo_per_file.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "modello", "K", "t1_s", "ritardo_ms", "falsi_allarmi_pre"])
        w.writeheader()
        w.writerows(righe)
    n_arco = len({r["file"] for r in righe})
    print(f"\nEsperimenti con arco: {n_arco}; senza arco (esclusi): {len(senza_arco)}; "
          f"nominale {args.nominale}; latenza di inferenza sommata: {args.latenza_ms:g} ms")
    print(f"{'Modello':<10}{'K':>3}{'mancati':>9}{'mediana':>10}{'95° perc.':>11}{'massimo':>10}   [ms]")
    riepilogo = []
    for k in args.modelli:
        for K in args.k:
            r = [x for x in righe if x["modello"] == k and x["K"] == K]
            d = np.array([x["ritardo_ms"] for x in r if x["ritardo_ms"] is not None])
            mancati = len(r) - len(d)
            riepilogo.append(dict(modello=k, K=K, n=len(r), mancati=mancati,
                                  mediana_ms=float(np.median(d)) if len(d) else None,
                                  p95_ms=float(np.percentile(d, 95)) if len(d) else None,
                                  massimo_ms=float(d.max()) if len(d) else None,
                                  esperimenti_con_falsi_allarmi_pre=int(sum(x["falsi_allarmi_pre"] > 0 for x in r))))
            if len(d):
                print(f"{k:<10}{K:>3}{mancati:>9}{np.median(d):>10.0f}{np.percentile(d, 95):>11.0f}{d.max():>10.0f}")
            else:
                print(f"{k:<10}{K:>3}{mancati:>9}   nessuna rilevazione")
    with open(out / "riepilogo_ritardo.json", "w", encoding="utf-8") as f:
        json.dump(dict(versione=VERSIONE, args={k: v for k, v in vars(args).items() if k != "mat_root"},
                       n_esperimenti_con_arco=n_arco, senza_arco=senza_arco, riepilogo=riepilogo), f, indent=1)
    print(f"\nSalvato in {out}/ (ritardo_per_file.csv e riepilogo_ritardo.json)")


if __name__ == "__main__":
    main()
