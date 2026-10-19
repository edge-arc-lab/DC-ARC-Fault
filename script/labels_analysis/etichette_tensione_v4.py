#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
etichette_tensione_v4.py - robustezza delle etichette dalla TENSIONE: parametri di t1_V, baseline "solo livello", finestre mancate
================================================================================================================================

Segue etichette_tensione_v3.py (che resta com'e'). Non riaddestra nulla; usa i quattro modelli gia' addestrati
e gli score in cache (--score-cache, o quelli gia' in --out). Tre analisi:

1. SWEEP DEI PARAMETRI di t1_V: soglia di scostamento (--dv-list, V) x durata di conferma (--confirm-list, ms),
   con persistenza e filtro fissi come nella v3. Per ogni combinazione: finestre, DR e FPR dei 4 modelli e DR
   nella fascia di livello >= 0.95. Dice se il quadro "i modelli non vedono gli archi vicini al nominale" regge
   al variare della definizione di t1_V.
2. BASELINE "SOLO LIVELLO" (una sola feature: media della finestra / nominale), con etichette dalla corrente
   e con etichette dalla tensione: soglia scelta su 4/5 degli esperimenti, valutata sul restante 1/5 (5 fold per
   esperimento, seed fisso). Con le etichette dalla corrente deve riprodurre il ~99.7% della tesi; con quelle
   dalla tensione mostra quanto quel valore dipenda dalla definizione.
3. FINESTRE MANCATE a livello >= 0.95 (caso base della v3): file, numero di finestre, t1 dalla corrente e dalla tensione,
   se il riferimento di tensione e' sporco. Scrive finestre_mancate_livello_alto.csv.

Filtro del riferimento (corregge quello della v3, che non segnalava nessun file): un file e' "rif. sporco" se piu' del
--max-frac-rif (default 5%) dei campioni filtrati dei primi 0.2 s si scosta dalla mediana di almeno la soglia (dv).
Le analisi 1 e 3 sono riportate su tutti i file e sui soli file con riferimento pulito.

Uso (dalla cartella della repo)
    python -B script/etichette_tensione_v4.py --repo . --mat-root "<cartella dei .mat>" --out ..\ris_etichette_V4 --score-cache ..\ris_etichette_V2\score_finestre.npz
    python -B script/etichette_tensione_v4.py --selftest
Richiede etichette_tensione_v3.py nella stessa cartella (o in una sottocartella di script/).
"""
import argparse
import csv
import importlib.util
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v4"
QUI = Path(__file__).resolve().parent


def carica_v3(repo):
    cand = []
    for base in (QUI, QUI.parent, repo / "script"):
        if base.is_dir():
            cand += list(base.rglob("etichette_tensione_v3.py"))
    if not cand:
        sys.exit("non trovo etichette_tensione_v3.py")
    sp = importlib.util.spec_from_file_location("et_v3", cand[0])
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


def rif_sporco(v, fs, base_s, dv, smooth_ms, max_frac):
    """True se il riferimento (primi base_s secondi) non e' stabile: troppi campioni lontani dalla mediana."""
    nb = max(10, int(round(base_s * fs)))
    b = v[:nb]
    r = float(np.median(b))
    bs = None
    from numpy.lib.stride_tricks import sliding_window_view
    n = max(1, int(round(smooth_ms / 1000 * fs)) | 1)
    bs = np.median(sliding_window_view(np.pad(b, n // 2, mode="edge"), n), axis=1) if n > 1 else b
    return float(np.mean(np.abs(bs - r) >= dv)) > max_frac


def prepara_v(v, fs, base_s, smooth_ms):
    """Filtro e riferimento calcolati UNA volta per file (indipendenti da soglia e conferma)."""
    from numpy.lib.stride_tricks import sliding_window_view
    nb = max(10, int(round(base_s * fs)))
    rif = float(np.median(v[:nb]))
    rum = 1.4826 * float(np.median(np.abs(v[:nb] - rif)))
    n = max(1, int(round(smooth_ms / 1000 * fs)) | 1)
    if n == 1:
        vs = v.copy()
    else:  # mediana mobile a blocchi (memoria limitata)
        xp = np.pad(v, n // 2, mode="edge")
        vs = np.empty(len(v))
        for a in range(0, len(v), 200000):
            vs[a:a + 200000] = np.median(sliding_window_view(xp[a:a + 200000 + n - 1], n), axis=1)
    return vs, rif, rum


def t1_da_prep(vs, rif, rum, fs, dv, k_mad, confirm_ms, persist_ms):
    """Stessa logica di etichette_tensione_v3.t1_da_tensione, su filtro/riferimento precalcolati."""
    soglia = max(dv, k_mad * rum)
    sopra = np.abs(vs - rif) >= soglia
    nc_ = max(1, int(round(confirm_ms / 1000 * fs)))
    npers = max(1, int(round(persist_ms / 1000 * fs)))
    if len(sopra) < nc_:
        return None
    c = np.convolve(sopra.astype(np.int32), np.ones(nc_, dtype=np.int32), mode="valid")
    for i in np.flatnonzero(c >= nc_):
        if abs(float(np.median(vs[i:i + npers])) - rif) >= soglia:
            return int(i)
    return None


def baseline_livello(y, L, grp, n_fold=5, seed=0):
    """Soglia sul livello scelta (massimizzando l'accuratezza bilanciata) su 4/5 degli esperimenti, valutata sul resto.
    Restituisce la predizione (1 = arco = livello sotto soglia) per ogni finestra."""
    esp = np.unique(grp)
    rng = np.random.default_rng(seed)
    rng.shuffle(esp)
    fold = {e: i % n_fold for i, e in enumerate(esp)}
    f = np.array([fold[g] for g in grp])
    pred = np.zeros(len(y), dtype=int)
    for k in range(n_fold):
        tr, te = f != k, f == k
        Lt, yt = L[tr], y[tr]
        o = np.argsort(Lt, kind="stable")
        Ls, ys = Lt[o], yt[o]
        na, npre = max(1, int((ys == 1).sum())), max(1, int((ys == 0).sum()))
        # predizione "arco" se L < t con t tra due valori consecutivi: tpr(i), tnr(i) con i campioni sotto soglia
        tpr = np.concatenate(([0], np.cumsum(ys == 1))) / na
        tnr = 1 - np.concatenate(([0], np.cumsum(ys == 0))) / npre
        i = int(np.argmax((tpr + tnr) / 2))
        bt = 0.95 if i == 0 else (Ls[i - 1] + (Ls[i] if i < len(Ls) else Ls[-1] + 1e-6)) / 2
        pred[te] = (L[te] < bt).astype(int)
    return pred


def selftest(et):
    fs = 10000
    rng = np.random.default_rng(0)
    t = np.arange(3 * fs) / fs
    v_ok = np.where(t < 0.5, 0.2, 50.0) + rng.normal(0, 0.3, len(t))
    v_sp = np.full(len(t), 13.0) + rng.normal(0, 0.3, len(t))
    v_sp[1500:2000] = 3.0                               # dip di 50 ms nei primi 200 ms (25%)
    vv = np.where(t < 0.5, 0.2, 50.0) + rng.normal(0, 0.3, len(t))
    vv[8000:8100] += 40
    for dv_, cf_ in ((5.0, 100.0), (10.0, 20.0), (3.0, 500.0)):
        vs_, r_, u_ = prepara_v(vv, fs, 0.2, 5.0)
        ok0 = t1_da_prep(vs_, r_, u_, fs, dv_, 10.0, cf_, 300.0) == et.t1_da_tensione(vv, fs, 0.2, dv_, 10.0, 5.0, cf_, 300.0)[0]
        if not ok0:
            print("t1_da_prep diverge da v3", dv_, cf_)
    ok = ok0 and (not rif_sporco(v_ok, fs, 0.2, 5.0, 5.0, 0.05)) and rif_sporco(v_sp, fs, 0.2, 5.0, 5.0, 0.05)
    # baseline: livello perfettamente separabile -> ~100%
    n = 400
    grp = np.array([f"E{i % 40:03d}" for i in range(n)])
    y = (np.arange(n) % 2).astype(int)
    L = np.where(y == 1, 0.8, 1.0) + rng.normal(0, 0.01, n)
    p = baseline_livello(y, L, grp)
    ok &= (p == y).mean() > 0.99
    # baseline: livello non informativo -> accuratezza bilanciata ~50%
    L2 = rng.normal(1.0, 0.01, n)
    p2 = baseline_livello(y, L2, grp)
    bal = (np.mean(p2[y == 1]) + np.mean(1 - p2[y == 0])) / 2
    ok &= 0.35 < bal < 0.65
    print(f"baseline separabile acc {(p == y).mean():.3f}; non informativo acc. bilanciata {bal:.3f}")
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_etichette_V4")
    ap.add_argument("--score-cache", default=None)
    ap.add_argument("--base-s", type=float, default=0.2)
    ap.add_argument("--dv", type=float, default=5.0)
    ap.add_argument("--k-mad", type=float, default=10.0)
    ap.add_argument("--smooth-ms", type=float, default=5.0)
    ap.add_argument("--confirm-ms", type=float, default=100.0)
    ap.add_argument("--persist-ms", type=float, default=300.0)
    ap.add_argument("--dv-list", type=float, nargs="+", default=[3.0, 5.0, 10.0, 20.0])
    ap.add_argument("--confirm-list", type=float, nargs="+", default=[20.0, 100.0, 200.0, 500.0])
    ap.add_argument("--max-frac-rif", type=float, default=0.05)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    et = carica_v3(repo)
    if args.selftest:
        sys.exit(selftest(et))
    if not args.mat_root:
        sys.exit("serve --mat-root oppure --selftest")
    s = et.importa_sens(repo)
    sys.path.insert(0, str(repo / "script"))
    try:
        import analisi_test_set_v1 as a
    except ImportError:
        import test_set_analysis as a
    P = s.P
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _orig = np.savez

    def _sicuro(file, *x, **k):
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        return _orig(file, *x, **k)

    np.savez = _sicuro
    if args.score_cache and Path(args.score_cache).exists() and not (out / "score_finestre.npz").exists():
        shutil.copy(args.score_cache, out / "score_finestre.npz")
    P(f"etichette_tensione {VERSIONE} - {time.strftime('%Y-%m-%d %H:%M:%S')}")

    with open(repo / a.TEST_META, newline="", encoding="utf-8") as f:
        meta = list(csv.DictReader(f))
    files = sorted({m["filename"] for m in meta})
    mats = s.indice_mat(args.mat_root)
    mancanti = [x for x in files if x not in mats]
    if mancanti:
        sys.exit(f"{len(mancanti)} file .mat non trovati, es. {mancanti[:3]}")
    istanze = s.chiavi_istanze(files, mats)
    from scipy.io import loadmat
    corr, tens = {}, {}
    for i, (k, path) in enumerate(istanze, 1):
        print(f"\r  lettura .mat {i}/{len(istanze)}", end="", flush=True)
        m = loadmat(path, squeeze_me=False)
        ch = {x.lower(): x for x in m if not x.startswith("__")}
        c = m[ch["currentdata"]].flatten().astype(np.float64)
        v = m[ch["voltagedata"]].flatten().astype(np.float64)
        n = min(len(c), len(v))
        corr[k], tens[k] = c[:n], v[:n]
    print()
    score_f = s.calcola_score_finestre(repo, a, corr, out)
    FS, WIN, STEP = s.FS, s.WIN_N, s.STEP_N
    chiavi = [k for k in sorted(corr) if len(corr[k]) >= WIN + STEP and float(np.median(corr[k][:s.NOMINAL_N])) >= s.BASE["min_nom"]]
    nom = {k: float(np.median(corr[k][:s.NOMINAL_N])) for k in chiavi}
    tI = {k: s.trova_t1(corr[k], nom[k], s.BASE["thresh"], s.BASE["confirm_ms"]) for k in chiavi}
    sporco = {k: rif_sporco(tens[k], FS, args.base_s, args.dv, args.smooth_ms, args.max_frac_rif) for k in chiavi}
    P(f"\nFile con riferimento di tensione sporco (> {args.max_frac_rif:.0%} dei campioni dei primi {args.base_s} s oltre {args.dv} V dalla mediana): "
      f"{sum(sporco.values())} su {len(chiavi)}")

    prep = {k: prepara_v(tens[k], FS, args.base_s, args.smooth_ms) for k in chiavi}

    def t1V_di(dv, conf):
        return {k: t1_da_prep(*prep[k], FS, dv, args.k_mad, conf, args.persist_ms) for k in chiavi}

    def costruisci(t1map, solo_puliti):
        Y, S, F, L, K = [], [], [], [], []
        for k in chiavi:
            if solo_puliti and sporco[k]:
                continue
            cur = corr[k]
            s0, s1 = s.seleziona(*s.finestre_etichettate(len(cur), t1map[k]))
            sc = score_f[k]
            for lab, lista in ((0, s0), (1, s1)):
                for st in lista:
                    Y.append(lab); S.append(sc[st // STEP]); F.append(s.nome_base(k))
                    L.append(et.livello_finestra(cur, nom[k], st, WIN)); K.append(k)
        return np.array(Y), np.array(S), np.array(F), np.array(L), np.array(K)

    def dr_fpr(y, S, j):
        pr = S[:, j] > 0
        return 100 * np.mean(pr[y == 1]) if (y == 1).any() else float("nan"), 100 * np.mean(pr[y == 0]) if (y == 0).any() else float("nan")

    riep = {"sporchi": int(sum(sporco.values()))}
    t1_base = t1V_di(args.dv, args.confirm_ms)

    # ---- 1. sweep
    P("\n== 1. Sweep dei parametri di t1_V (etichette dalla tensione; modelli non riaddestrati) ==")
    P("DR % dei 4 modelli (Inc FP32 / Inc INT8 / MS FP32 / MS INT8) e DR nella fascia di livello >= 0.95 (n finestre; DR Inc FP32)")
    righe_sw = []
    for solo in (False, True):
        P(f"\n-- {'solo file con riferimento pulito' if solo else 'tutti i file'} --")
        P(f"{'dv [V]':>7}{'conferma [ms]':>14}{'finestre arco/pre':>20}{'DR % (4 modelli)':>30}{'FPR % (max)':>13}{'fascia>=0.95: n, DR IncFP32':>30}")
        for dv in args.dv_list:
            for cf in args.confirm_list:
                t1m = t1V_di(dv, cf)
                senza = [k for k in chiavi if t1m[k] is None]
                tm = {k: t1m[k] for k in chiavi}
                y, S, fl, L, K = costruisci(tm, solo)
                drs = [dr_fpr(y, S, j) for j in range(4)]
                alto = (y == 1) & (L >= 0.95)
                dr_alto = 100 * np.mean(S[alto, 0] > 0) if alto.any() else float("nan")
                P(f"{dv:>7g}{cf:>14g}{f'{int((y == 1).sum())}/{int((y == 0).sum())}':>20}"
                  f"{' / '.join(f'{d[0]:.1f}' for d in drs):>30}{max(d[1] for d in drs):>13.1f}{f'{int(alto.sum())}, {dr_alto:.1f}':>30}")
                righe_sw.append([int(solo), dv, cf, int((y == 1).sum()), int((y == 0).sum()), *[f"{d[0]:.2f}" for d in drs],
                                 *[f"{d[1]:.2f}" for d in drs], int(alto.sum()), f"{dr_alto:.2f}", len(senza)])
    with open(out / "sweep_t1V.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["solo_puliti", "dv_V", "conferma_ms", "n_arco", "n_pre", "DR_incFP32", "DR_incINT8", "DR_msFP32", "DR_msINT8",
                    "FPR_incFP32", "FPR_incINT8", "FPR_msFP32", "FPR_msINT8", "n_arco_livello_alto", "DR_incFP32_livello_alto", "file_senza_t1V"])
        w.writerows(righe_sw)

    # ---- 2. baseline solo livello
    P("\n== 2. Baseline 'solo livello' (soglia sul livello, 5 fold per esperimento) ==")
    P(f"{'Etichette':<12}{'Accuratezza %':>15}{'DR %':>22}{'FPR %':>22}{'| modelli: DR (Inc FP32)':>26}")
    import re
    base = {}
    for nome, t1m in (("corrente", tI), ("tensione", t1_base)):
        y, S, fl, L, K = costruisci(t1m, False)
        grp = np.array([re.match(r"^(.*?)_Study\d+", f).group(1) for f in fl])
        pred = baseline_livello(y, L, grp, seed=args.seed)
        acc = 100 * np.mean(pred == y)
        dr = et.boot_rate(pred[y == 1] == 1, fl[y == 1], args.n_boot, args.seed)
        fpr_ok = et.boot_rate(pred[y == 0] == 0, fl[y == 0], args.n_boot, args.seed)
        fpr = (100 - fpr_ok[0], 100 - fpr_ok[2], 100 - fpr_ok[1])
        d_inc = dr_fpr(y, S, 0)[0]
        P(f"{nome:<12}{acc:>15.1f}{f'{dr[0]:.1f} [{dr[1]:.1f}, {dr[2]:.1f}]':>22}{f'{fpr[0]:.1f} [{fpr[1]:.1f}, {fpr[2]:.1f}]':>22}{d_inc:>26.1f}")
        base[nome] = dict(acc=acc, DR=dr[:3], FPR=fpr, n=int(len(y)), DR_incFP32=d_inc)
    riep["baseline_solo_livello"] = base

    # ---- 3. finestre mancate a livello alto
    P("\n== 3. Finestre di arco (secondo la tensione) con livello >= 0.95 (caso base) ==")
    y, S, fl, L, K = costruisci(t1_base, False)
    alto = np.flatnonzero((y == 1) & (L >= 0.95))
    per_file = {}
    for i in alto:
        d = per_file.setdefault(K[i], dict(n=0, mancate_incFP32=0, mancate_tutti=0))
        d["n"] += 1
        d["mancate_incFP32"] += int(S[i, 0] <= 0)
        d["mancate_tutti"] += int(all(S[i, j] <= 0 for j in range(4)))
    righe = []
    for k, d in sorted(per_file.items(), key=lambda x: -x[1]["n"]):
        righe.append([k, d["n"], d["mancate_incFP32"], d["mancate_tutti"], (tI[k] / FS if tI[k] is not None else ""),
                      (t1_base[k] / FS if t1_base[k] is not None else ""), int(sporco[k]), f"{nom[k]:.3f}"])
    with open(out / "finestre_mancate_livello_alto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "n_finestre", "mancate_IncFP32", "mancate_tutti_i_modelli", "t1_corrente_s", "t1_tensione_s", "rif_sporco", "nominale_A"])
        w.writerows(righe)
    P(f"{len(alto)} finestre in {len(per_file)} file; di cui in file con riferimento sporco: {sum(d['n'] for k, d in per_file.items() if sporco[k])}; "
      f"mancate da tutti e 4 i modelli: {sum(d['mancate_tutti'] for d in per_file.values())}")
    P(f"{'file':<62}{'n':>4}{'t1 corr [s]':>13}{'t1 tens [s]':>13}{'rif sporco':>12}")
    for r in righe[:12]:
        P(f"{r[0][:60]:<62}{r[1]:>4}{(f'{r[4]:.3f}' if r[4] != '' else '-'):>13}{(f'{r[5]:.3f}' if r[5] != '' else '-'):>13}{('si' if r[6] else 'no'):>12}")
    riep["livello_alto"] = dict(n_finestre=int(len(alto)), n_file=len(per_file),
                                in_file_rif_sporco=int(sum(d["n"] for k, d in per_file.items() if sporco[k])))
    (out / "riepilogo_etichette_v4.json").write_text(json.dumps(riep, indent=1, default=float), encoding="utf-8")
    P("\nLimiti: t1_V resta euristico; i modelli sono addestrati con le etichette dalla corrente; baseline con una sola feature (livello medio) e soglia unica; "
      "una sola esecuzione sul test set.")
    (out / "report.txt").write_text("\n".join(s.LOG) + "\n", encoding="utf-8")
    P(f"\nFile scritti in: {out}")


if __name__ == "__main__":
    main()
