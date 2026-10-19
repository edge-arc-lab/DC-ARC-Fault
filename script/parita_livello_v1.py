#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
parita_livello_v1.py - i modelli aggiungono informazione oltre al livello di corrente?
======================================================================================

Per le finestre del TEST SET pubblicato (7161) confronta i quattro modelli con un classificatore "solo livello"
(livello = media della finestra / nominale: le finestre sono gia' normalizzate, quindi livello = X.mean) a PARITA' DI LIVELLO:
le finestre sono divise in fasce di livello e in ogni fascia si misura quanto bene arco e pre-arco sono separati
(a) dallo score dei modelli e (b) dal solo livello. Se in una fascia stretta il livello non separa piu' le classi (AUC ~ 0.5) ma
i modelli si', i modelli usano qualcosa oltre al livello (forma d'onda, rumore); se anche i modelli sono a ~0.5, non c'e'
informazione oltre al livello. Si ripete per due definizioni di etichetta:
  - "corrente": etichette del test set pubblicato (t1 dalla corrente);
  - "tensione": t1_V da etichette_tensione_v3.csv (colonna t1_V_s): finestra di arco se start >= t1_V, pre-arco se end <= t1_V,
    scartata se a cavallo.

Per ogni fascia e schema: n finestre arco / pre-arco, AUC dei modelli e AUC del solo livello (con IC 95% bootstrap per esperimento).
AUC = probabilita' che una finestra di arco abbia score piu' "da arco" di una di pre-arco (0.5 = nessuna separazione).
Gli score dei modelli si calcolano con i modelli ONNX della repo (cache in --out/score_test.npz).

Uso (dalla cartella della repo)
    python -B script/parita_livello_v1.py --repo . --t1-csv ..\ris_etichette_V2\etichette_tensione_v3.csv --out ..\ris_parita
    python -B script/parita_livello_v1.py --selftest
Richiede test_set_analysis.py (o analisi_test_set_v1.py) e i modelli ONNX nella repo.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v1"
QUI = Path(__file__).resolve().parent
FASCE = [(0.0, 0.90), (0.90, 0.94), (0.94, 0.96), (0.96, 0.98), (0.98, 1.00), (1.00, 9.0)]
MODELLI = ["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"]
NOMI = {"inc_fp32": "Inc FP32", "inc_int8": "Inc INT8", "ms_fp32": "MS FP32", "ms_int8": "MS INT8"}


def auc(y, s):
    """AUC (Mann-Whitney) con i ranghi medi per i pareggi; nan se manca una classe."""
    from scipy.stats import rankdata
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_ci(y, s, grp, n_boot, seed):
    """AUC con IC 95% bootstrap per esperimento (ricampiona i gruppi)."""
    a = auc(y, s)
    if np.isnan(a):
        return a, float("nan"), float("nan")
    esp, cod = np.unique(grp, return_inverse=True)
    rng = np.random.default_rng(seed)
    idx_per = [np.flatnonzero(cod == i) for i in range(len(esp))]
    v = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(esp), len(esp))
        ii = np.concatenate([idx_per[p] for p in pick])
        x = auc(y[ii], s[ii])
        if not np.isnan(x):
            v.append(x)
    if len(v) < 10:
        return a, float("nan"), float("nan")
    return a, float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def etichette_tensione(meta, t1csv):
    """Etichetta 0/1 per finestra (-1 = scartata) da t1_V; sceglie l'istanza il cui t1_I e' piu' vicino a quello dei metadati."""
    inst = {}
    with open(t1csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            nome = r["file"].split("#")[0]
            if r["t1_V_s"] == "" or r["t1_I_s"] == "":
                continue
            inst.setdefault(nome, []).append((float(r["t1_I_s"]), float(r["t1_V_s"])))
    lab = np.full(len(meta), -1, dtype=int)
    for i, m in enumerate(meta):
        c = inst.get(m["filename"])
        if not c:
            continue
        t1m = float(m["t1_s"])
        tV = min(c, key=lambda x: abs(x[0] - t1m))[1]
        a, b = float(m["t_start_s"]), float(m["t_end_s"])
        lab[i] = 1 if a >= tV else (0 if b <= tV else -1)
    return lab


def selftest():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)].astype(int)
    ok = abs(auc(y, y + rng.normal(0, 0.1, 400)) - 1.0) < 0.01
    ok &= abs(auc(y, rng.normal(0, 1, 400)) - 0.5) < 0.1
    ok &= abs(auc(y, np.zeros(400)) - 0.5) < 1e-9           # tutti pareggi
    grp = np.array([f"E{i % 20}" for i in range(400)])
    a, lo, hi = auc_ci(y, y + rng.normal(0, 0.1, 400), grp, 50, 1)
    ok &= lo <= a <= hi
    meta = [dict(filename="f", t1_s="1.0", t_start_s=str(s), t_end_s=str(s + 0.1)) for s in (0.0, 0.35, 0.45, 0.9)]
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "t.csv")
        open(p, "w").write("file,t1_I_s,t1_V_s\nf,1.0,0.4\n")
        lab = etichette_tensione(meta, p)
    ok &= list(lab) == [0, -1, 1, 1]
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--t1-csv", default=None, help="etichette_tensione_v3.csv (colonne file, t1_I_s, t1_V_s)")
    ap.add_argument("--out", default="ris_parita")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if not a.t1_csv:
        sys.exit("serve --t1-csv (o --selftest)")
    repo = Path(a.repo) if a.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    sys.path.insert(0, str(repo / "script"))
    try:
        import analisi_test_set_v1 as t
    except ImportError:
        import test_set_analysis as t
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y_cur, chiavi, idx = t.carica_test(repo, None, 0)
    with open(repo / t.TEST_META, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    cache = out / "score_test.npz"
    if cache.exists() and len(np.load(cache)["inc_fp32"]) == len(y_cur):
        sc = {k: np.load(cache)[k] for k in MODELLI}
        print("score dalla cache", cache)
    else:
        sc, _ = t.calcola_score(repo, X, idx, out, False)
        np.savez(cache, **{k: sc[k] for k in MODELLI})
    liv = X.mean(axis=1).astype(np.float64)
    y_ten = etichette_tensione(meta, a.t1_csv)
    righe, riep = [], {}
    for nome, y in (("corrente", y_cur), ("tensione", y_ten)):
        ok_ = y >= 0
        print(f"\n=== etichette dalla {nome}: {int(ok_.sum())} finestre ({int((y[ok_] == 1).sum())} arco, {int((y[ok_] == 0).sum())} pre-arco) ===")
        print(f"{'fascia di livello':<18}{'arco/pre':>12}" + "".join(f"{NOMI[k]:>20}" for k in MODELLI) + f"{'solo livello':>20}")
        for lo_, hi_ in [(-1, 99)] + FASCE:
            m = ok_ & (liv >= lo_) & (liv < hi_) if lo_ >= 0 else ok_
            yy, g = y[m], chiavi[m]
            etich = "tutte" if lo_ < 0 else f"{lo_:.2f}-{hi_:.2f}" if hi_ < 9 else f">={lo_:.2f}"
            cells = []
            for k in MODELLI + ["livello"]:
                s = -liv[m] if k == "livello" else sc[k][m]
                am, lo, hi = auc_ci(yy, s, g, a.n_boot, a.seed) if min((yy == 1).sum(), (yy == 0).sum()) >= 20 else (auc(yy, s), float("nan"), float("nan"))
                cells.append(am)
                righe.append([nome, etich, int((yy == 1).sum()), int((yy == 0).sum()), k, f"{am:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
                riep.setdefault(nome, {}).setdefault(etich, {})[k] = dict(auc=am, lo=lo, hi=hi, n_arco=int((yy == 1).sum()), n_pre=int((yy == 0).sum()))
            print(f"{etich:<18}{f'{int((yy == 1).sum())}/{int((yy == 0).sum())}':>12}" + "".join(f"{c:>20.3f}" for c in cells))
    with open(out / "auc_per_fascia.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["etichette", "fascia", "n_arco", "n_pre", "score", "AUC", "AUC_lo", "AUC_hi"])
        w.writerows(righe)
    (out / "riepilogo_parita.json").write_text(json.dumps(riep, indent=1, default=float), encoding="utf-8")
    print(f"\nScritto in {out.resolve()}")


if __name__ == "__main__":
    main()
