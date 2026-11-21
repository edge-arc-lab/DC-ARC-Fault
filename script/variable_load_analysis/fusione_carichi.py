#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulazione di transitori reali tramite concatenazione di registrazioni (Carico A -> Carico B).

Valuta la risposta della stima causale I_nom(t) e dei classificatori a gradini di carico reali:
- Calo forte (B/A < 0.70)
- Calo lieve (0.70 - 0.95)
- Aumento lieve (1.05 - 1.50)
- Aumento forte (> 1.50)
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[1] if (QUI.parents[1] / "dataset").is_dir() else QUI.parent
for p in [str(QUI), str(REPO / "script"), str(REPO / "script" / "casual_current_I_norm"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import norm_causale as nc
import test_set_analysis as a

CLASSI = [("calo forte (B/A < 0.70)", 0.0, 0.70), ("calo lieve (0.70-0.95)", 0.70, 0.95),
          ("aumento lieve (1.05-1.50)", 1.05, 1.50), ("aumento forte (> 1.50)", 1.50, 99.0)]


def fusione(raw_a, raw_b, p, pad_s, W, gate_frac):
    nom_a = float(np.median(raw_a[:p.NOMINAL_N]))
    t1b, nom_b = p.find_t1(raw_b)
    if t1b is None or nom_a < p.MIN_NOM_A:
        return None
    pad_n = int(round(pad_s * p.FS_HZ / p.STEP_N)) * p.STEP_N
    reps = int(np.ceil(pad_n / p.NOMINAL_N))
    sig = np.concatenate([np.tile(raw_a[:p.NOMINAL_N], reps)[:pad_n], raw_b])
    stima = {s: (i, acc) for s, i, acc in nc.nominale_causale(sig, W, "level", p, None, gate_frac)}
    righe = []
    for s in range(0, len(sig) - p.WIN_N + 1, p.STEP_N):
        if s + p.WIN_N <= pad_n:
            gruppo, lab = "regime_A", 0
        elif s >= pad_n:
            t1 = t1b + pad_n
            if s >= t1:
                gruppo, lab = "B", 1
            elif s + p.WIN_N <= t1:
                gruppo, lab = "B", 0
            else:
                continue
        else:
            continue
        w = sig[s:s + p.WIN_N]
        righe.append((gruppo, lab, w, stima[s][0]))
    if not righe:
        return None
    gruppi = np.array([r[0] for r in righe])
    lab = np.array([r[1] for r in righe])
    W_ = np.stack([r[2] for r in righe])
    i_caus = np.array([r[3] for r in righe])
    ref = W_ / nom_b
    attivo = (lab == 1) & (ref.mean(1) >= 0.75)
    return dict(nom_a=nom_a, nom_b=nom_b, gruppo=gruppi, lab=lab, attivo=attivo,
                norm={"statico": (W_ / nom_a).astype(np.float32),
                      "causale": (W_ / i_caus[:, None]).astype(np.float32),
                      "riferimento": ref.astype(np.float32)})


def classifica(norm, soglia, punteggi):
    out = {"soglia": (norm.mean(1) < soglia)}
    for k, f in punteggi.items():
        out[k] = f(norm) > 0
    return out


def tabella(dati, soglia, punteggi, nomi):
    acc = {}
    for d in dati:
        r = d["nom_b"] / d["nom_a"]
        cl = next((c[0] for c in CLASSI if c[1] <= r < c[2]), None)
        if cl is None:
            continue
        for nm, X in d["norm"].items():
            for k, pred in classifica(X, soglia, punteggi).items():
                a = acc.setdefault((cl, nm, k), dict(fp=0, n_pre=0, tp=0, n_arc=0, fpA=0, n_A=0))
                m_pre = (d["gruppo"] == "B") & (d["lab"] == 0)
                m_arc = d["attivo"]
                m_A = (d["gruppo"] == "regime_A") & (nm != "riferimento")
                a["fp"] += int((pred & m_pre).sum()); a["n_pre"] += int(m_pre.sum())
                a["tp"] += int((pred & m_arc).sum()); a["n_arc"] += int(m_arc.sum())
                a["fpA"] += int((pred & m_A).sum()); a["n_A"] += int(m_A.sum())
    righe = []
    for (cl, nm, k), a in sorted(acc.items(), key=lambda x: ([c[0] for c in CLASSI].index(x[0][0]), x[0][1], x[0][2])):
        righe.append(dict(classe=cl, normalizzazione=nm, classificatore=nomi.get(k, k),
                          FPR_B_pre=100 * a["fp"] / max(a["n_pre"], 1), DR_arco_attivo=100 * a["tp"] / max(a["n_arc"], 1),
                          FPR_regime_A=(100 * a["fpA"] / a["n_A"]) if a["n_A"] else float("nan"), n_pre=a["n_pre"], n_arco=a["n_arc"]))
    return righe


def stampa(righe, out):
    cl_prec = None
    for r in righe:
        if r["classe"] != cl_prec:
            print(f"\n=== {r['classe']}")
            print(f"  {'normalizzazione':<16}{'classificatore':<26}{'FPR (carico cambia, no arco)':>30}{'DR arco attivo':>16}{'FPR regime A':>14}")
            cl_prec = r["classe"]
        print(f"  {r['normalizzazione']:<16}{r['classificatore']:<26}{r['FPR_B_pre']:>29.1f}%{r['DR_arco_attivo']:>15.1f}%{('-' if r['FPR_regime_A'] != r['FPR_regime_A'] else format(r['FPR_regime_A'], '.1f') + '%'):>14}")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        with open(Path(out) / "fusione_carichi.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
            w.writeheader(); w.writerows(righe)
        print(f"\nSalvato {Path(out) / 'fusione_carichi.csv'}")


def sintetico(rng, p, A, durata=3.0, t_arco=0.5, livello=0.88):
    n = int(durata * p.FS_HZ)
    x = A * (1 + 0.004 * rng.standard_normal(n))
    k = int(t_arco * p.FS_HZ)
    x[k:] = A * livello * (1 + 0.02 * rng.standard_normal(n - k))
    return x


def selftest(p):
    rng = np.random.default_rng(0)
    dati = []
    for A, B in [(5.0, 2.0), (4.0, 1.5), (2.0, 5.0), (1.5, 4.0)]:
        d = fusione(sintetico(rng, p, A), sintetico(rng, p, B), p, 4.0, 2.0, 0.98)
        dati.append(d)
    righe = tabella(dati, 0.9413, {}, {"soglia": "soglia sul livello"})
    t = {(r["classe"], r["normalizzazione"]): r for r in righe}
    ok = True
    for cl in ("calo forte (B/A < 0.70)", "aumento forte (> 1.50)"):
        ref = t[(cl, "riferimento")]
        ok &= ref["FPR_B_pre"] < 1 and ref["DR_arco_attivo"] > 99
    ok &= t[("calo forte (B/A < 0.70)", "statico")]["FPR_B_pre"] > 90
    ok &= t[("aumento forte (> 1.50)", "statico")]["DR_arco_attivo"] < 5
    stampa(righe, None)
    print("\nselftest", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Fonde due registrazioni di test con carico diverso e confronta soglia sul livello e modelli")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_fusione")
    ap.add_argument("--coppie", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dmin-A", type=float, default=0.5)
    ap.add_argument("--soglia", type=float, default=0.9413)
    ap.add_argument("--W", type=float, default=2.0)
    ap.add_argument("--gate-frac", type=float, default=0.98)
    ap.add_argument("--pad", type=float, default=4.0)
    ap.add_argument("--modelli", nargs="+", default=["inc_fp32", "ms_fp32"], choices=["inc_fp32", "inc_int8", "ms_fp32", "ms_int8"])
    ap.add_argument("--downsample-factor", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo) if args.repo else REPO
    p = nc.carica_parametri(repo)
    if args.selftest:
        sys.exit(selftest(p))
    if not args.mat_root:
        sys.exit("serve --mat-root (cartella dei .mat) oppure --selftest")

    from scipy.io import loadmat

    with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    nominali = {}
    for m in meta:
        nominali.setdefault(m["filename"], float(m["nominal_A"]))
    mats = nc.indice_mat(args.mat_root)
    files = sorted(fn for fn in nominali if fn in mats)
    if not files:
        sys.exit(f"nessun .mat del test set trovato in {args.mat_root}")

    sess = {k: a.apri_sessione(repo / a.MODELLI[k]) for k in args.modelli}
    coef = np.load(repo / a.RIDGE_COEF).astype(np.float64).reshape(-1)
    itc = float(np.load(repo / a.RIDGE_INTERCEPT)).reshape(-1)[0]
    punteggi = {k: (lambda X, k=k: a.score_inception(sess[k], X) if k.startswith("inc")
                    else a.score_msrcfe(sess[k], X, coef, itc)) for k in args.modelli}
    nomi = {"soglia": "soglia sul livello", **a.ETICHETTE}

    rng = np.random.default_rng(args.seed)
    cache = {}

    def leggi(fn):
        if fn not in cache:
            cache[fn] = loadmat(mats[fn], squeeze_me=False)["CurrentData"].flatten()[::args.downsample_factor].astype(np.float64)
        return cache[fn]

    dati, tentativi = [], 0
    per_classe = {c[0]: 0 for c in CLASSI}
    quota = int(np.ceil(args.coppie / len(CLASSI)))
    while sum(per_classe.values()) < args.coppie and tentativi < 50 * args.coppie:
        tentativi += 1
        fa, fb = rng.choice(files, 2, replace=False)
        r = nominali[fb] / nominali[fa]
        if abs(nominali[fb] - nominali[fa]) < args.dmin_A:
            continue
        cl = next((c[0] for c in CLASSI if c[1] <= r < c[2]), None)
        if cl is None or per_classe[cl] >= quota:
            continue
        d = fusione(leggi(fa), leggi(fb), p, args.pad, args.W, args.gate_frac)
        if d is None:
            continue
        dati.append(d)
        per_classe[cl] += 1
    print(f"coppie usate: {len(dati)}  per classe: {per_classe}")
    stampa(tabella(dati, args.soglia, punteggi, nomi), args.out)


if __name__ == "__main__":
    main()
