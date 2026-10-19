#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
etichette_tensione_v3.py - valutazione dei modelli con etichette ricavate dalla TENSIONE, per fasce di livello
===============================================================================================================

Passo 2 del piano (v3: percorso di output assoluto; il salvataggio della cache ricrea la cartella se manca) "etichette indipendenti dal livello". Non riaddestra nulla: usa i quattro modelli GIA'
addestrati sul test set e confronta due modi di etichettare le stesse finestre da 100 ms (passo 50 ms,
max 10 per classe e per file, seed 42, finestre a cavallo di t1 scartate - come la pipeline):
  - "corrente" : t1 dalla corrente (caso base della tesi: < 0.95 x nominale per 300 ms);
  - "tensione" : t1_V dalla tensione ai capi degli elettrodi (VoltageData).

t1_V (v2, piu' robusto della v1):
  - riferimento = mediana di VoltageData nei primi --base-s s (0.2); rumore = 1.4826*MAD;
  - soglia = max(--dv, --k-mad * rumore);
  - VoltageData filtrata con mediana mobile --smooth-ms (5 ms);
  - candidato = primo istante in cui |v - rif| >= soglia per --confirm-ms (100 ms, v1: 20 ms);
  - PERSISTENZA: la mediana della tensione nei --persist-ms (300 ms) successivi al candidato deve
    scostarsi ancora dal riferimento di almeno la soglia (scarta le oscillazioni che rientrano);
  - i file il cui riferimento e' instabile (rumore nel riferimento > --max-rum-ref V, default 2) sono
    marcati "rif. instabile" e riportati anche a parte: l'analisi si fa sia su tutti i file sia su quelli stabili.

Valutazione: per le finestre di ogni schema DR e FPR dei quattro modelli (IC 95% bootstrap per esperimento) e DR per FASCIA DI LIVELLO
(livello = media della finestra / nominale: < 0.90, 0.90-0.95, >= 0.95) sulle finestre di arco secondo la tensione.
Il caso "tensione, solo livello >= 0.95" e' quello che la definizione sul livello non puo' vedere come arco.

Uso (dalla cartella della repo)
    python -B script/etichette_tensione_v3.py --repo . --mat-root "<cartella dei .mat>" --out ..\ris_etichette_V2 --score-cache ..\ris_sensibilita\score_finestre.npz
    python -B script/etichette_tensione_v3.py --selftest
--score-cache e' opzionale: se punta al score_finestre.npz della sensibilita' (stessa istanza di .mat) si salta l'inferenza
(altrimenti 10-40 min); la cache viene copiata in --out e completata se mancano istanze.
Richiede lab_analysis_2.0.py (o sensibilita_etichette_v2.py), test_set_analysis.py (o analisi_test_set_v1.py), i modelli ONNX nella repo.
Scrive etichette_tensione_v3.csv (t1 per file), valutazione_etichette.csv, riepilogo_etichette_v3.json, report.txt.
"""
import argparse
import csv
import importlib.util
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

VERSIONE = "v3"
QUI = Path(__file__).resolve().parent


def importa_sens(repo):
    """Importa lo script di sensibilita' (v2) da repo/script/** o dalla cartella di questo file."""
    cand = []
    for base in (QUI, QUI.parent, repo / "script"):
        if base.is_dir():
            cand += list(base.rglob("lab_analysis_2.0.py")) + list(base.rglob("sensibilita_etichette_v2.py"))
    if not cand:
        sys.exit("non trovo lab_analysis_2.0.py / sensibilita_etichette_v2.py")
    spec = importlib.util.spec_from_file_location("sens_v2", cand[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mediana_mobile(x, n):
    n = max(1, int(n) | 1)
    if n == 1:
        return x.copy()
    from numpy.lib.stride_tricks import sliding_window_view
    xp = np.pad(x, n // 2, mode="edge")
    return np.median(sliding_window_view(xp, n), axis=1)


def t1_da_tensione(v, fs, base_s=0.2, dv=5.0, k_mad=10.0, smooth_ms=5.0, confirm_ms=100.0, persist_ms=300.0):
    """(indice t1_V o None, v_rif, soglia, rumore_rif)."""
    nb = max(10, int(round(base_s * fs)))
    base = v[:nb]
    rif = float(np.median(base))
    rum = 1.4826 * float(np.median(np.abs(base - rif)))
    soglia = max(dv, k_mad * rum)
    vs = mediana_mobile(v, int(round(smooth_ms / 1000 * fs)))
    sopra = np.abs(vs - rif) >= soglia
    nc_ = max(1, int(round(confirm_ms / 1000 * fs)))
    npers = max(1, int(round(persist_ms / 1000 * fs)))
    if len(sopra) < nc_:
        return None, rif, soglia, rum
    c = np.convolve(sopra.astype(np.int32), np.ones(nc_, dtype=np.int32), mode="valid")
    for i in np.flatnonzero(c >= nc_):
        if abs(float(np.median(vs[i:i + npers])) - rif) >= soglia:
            return int(i), rif, soglia, rum
    return None, rif, soglia, rum


def livello_finestra(cur, nom, s, win):
    return float(np.mean(cur[s:s + win]) / nom)


def fascia(l):
    return "<0.90" if l < 0.90 else ("0.90-0.95" if l < 0.95 else ">=0.95")


FASCE = ["<0.90", "0.90-0.95", ">=0.95"]


def boot_rate(corretto, files, n_boot, seed):
    """Percentuale di 'corretto' con IC 95% bootstrap per esperimento."""
    n = len(corretto)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    chiavi = np.array([re.match(r"^(.*?)_Study\d+", f).group(1) for f in files])
    esp, cod = np.unique(chiavi, return_inverse=True)
    k = np.bincount(cod, weights=corretto.astype(float), minlength=len(esp))
    t = np.bincount(cod, minlength=len(esp)).astype(float)
    W = np.random.default_rng(seed).multinomial(len(esp), np.full(len(esp), 1.0 / len(esp)), size=n_boot).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = (W @ k) / (W @ t)
    lo, hi = np.nanpercentile(r, [2.5, 97.5]) * 100
    return 100 * k.sum() / t.sum(), lo, hi, n


def selftest():
    fs = 10000
    rng = np.random.default_rng(0)
    t = np.arange(3 * fs) / fs
    ok = True
    v = np.where(t < 0.5, 0.2, 50.0) + rng.normal(0, 0.3, len(t))
    r = t1_da_tensione(v, fs)
    ok &= r[0] is not None and abs(r[0] / fs - 0.5) < 0.01
    # oscillazione che rientra (come i dip pre-arco): non e' un innesco
    v2 = np.full(len(t), 13.0) + rng.normal(0, 0.3, len(t))
    v2[1500:2400] = 3.0
    ok &= t1_da_tensione(v2, fs)[0] is None
    # dip nei primi 0.2 s poi salita vera
    v3 = np.where(t < 0.25, 13.0, 28.0) + rng.normal(0, 0.3, len(t))
    v3[1500:2400] = 3.0
    r3 = t1_da_tensione(v3, fs)
    ok &= r3[0] is not None and r3[0] / fs > 0.2
    # boot_rate
    f = np.array([f"E{i%4:03d}_x_Study001_y" for i in range(40)])
    p, lo, hi, n = boot_rate(np.ones(40), f, 200, 1)
    ok &= p == 100 and lo == 100 and n == 40
    ok &= [fascia(x) for x in (0.8, 0.92, 0.96)] == FASCE
    print("SELFTEST", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_etichette_V2")
    ap.add_argument("--score-cache", default=None)
    ap.add_argument("--base-s", type=float, default=0.2)
    ap.add_argument("--dv", type=float, default=5.0)
    ap.add_argument("--k-mad", type=float, default=10.0)
    ap.add_argument("--smooth-ms", type=float, default=5.0)
    ap.add_argument("--confirm-ms", type=float, default=100.0)
    ap.add_argument("--persist-ms", type=float, default=300.0)
    ap.add_argument("--max-rum-ref", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(selftest())
    if not args.mat_root:
        sys.exit("serve --mat-root oppure --selftest")

    repo = Path(args.repo) if args.repo else (QUI.parent if (QUI.parent / "dataset").is_dir() else Path.cwd())
    s = importa_sens(repo)
    sys.path.insert(0, str(repo / "script"))
    try:
        import analisi_test_set_v1 as a
    except ImportError:
        import test_set_analysis as a
    P = s.P
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    _orig_savez = np.savez

    def _savez_sicuro(file, *a, **k):
        # la cache si scrive dopo ore di inferenza: se la cartella e' sparita nel frattempo, la ricrea
        Path(file).parent.mkdir(parents=True, exist_ok=True)
        return _orig_savez(file, *a, **k)

    np.savez = _savez_sicuro
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
    t1V, instabile, righe_t1 = {}, {}, []
    for k in sorted(corr):
        cur = corr[k]
        if len(cur) < WIN + STEP:
            continue
        nom = float(np.median(cur[:s.NOMINAL_N]))
        if nom < s.BASE["min_nom"]:
            continue
        tI = s.trova_t1(cur, nom, s.BASE["thresh"], s.BASE["confirm_ms"])
        tV, rif, sog, rum = t1_da_tensione(tens[k], FS, args.base_s, args.dv, args.k_mad, args.smooth_ms,
                                           args.confirm_ms, args.persist_ms)
        t1V[k] = (tI, tV)
        instabile[k] = rum > args.max_rum_ref
        righe_t1.append([k, tI / FS if tI is not None else "", tV / FS if tV is not None else "", rif, sog, rum,
                         int(instabile[k]), (tI - tV) / FS * 1000 if tI is not None and tV is not None else ""])
    with open(out / "etichette_tensione_v3.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "t1_I_s", "t1_V_s", "v_rif", "soglia_V", "rumore_rif_V", "rif_instabile", "diff_ms"])
        w.writerows(righe_t1)
    d = np.array([r[7] for r in righe_t1 if r[7] != ""], dtype=float)
    P(f"\nFile con riferimento di tensione instabile (rumore > {args.max_rum_ref} V): {sum(instabile.values())} su {len(instabile)}")
    P(f"File senza t1_V: {sum(1 for k in t1V if t1V[k][1] is None)}; senza t1 corrente: {sum(1 for k in t1V if t1V[k][0] is None)}")
    if len(d):
        q = np.percentile(d, [5, 25, 50, 75, 95])
        P(f"t1_I - t1_V [ms]: 5° {q[0]:.0f}  25° {q[1]:.0f}  mediana {q[2]:.0f}  75° {q[3]:.0f}  95° {q[4]:.0f}  min {d.min():.0f}  max {d.max():.0f}")

    def costruisci(schema, solo_stabili):
        Y, S, F, L = [], [], [], []
        for k in sorted(t1V):
            if solo_stabili and instabile[k]:
                continue
            tI, tV = t1V[k]
            t1 = tI if schema == "corrente" else tV
            cur = corr[k]
            nom = float(np.median(cur[:s.NOMINAL_N]))
            s0, s1 = s.seleziona(*s.finestre_etichettate(len(cur), t1))
            sc = score_f[k]
            for lab, lista in ((0, s0), (1, s1)):
                for st in lista:
                    Y.append(lab)
                    S.append(sc[st // STEP])
                    F.append(s.nome_base(k))
                    L.append(livello_finestra(cur, nom, st, WIN))
        return np.array(Y), np.array(S), np.array(F), np.array(L)

    righe, riep = [], {}
    for solo_stab in (False, True):
        tag = "file stabili" if solo_stab else "tutti i file"
        for schema in ("corrente", "tensione"):
            y, S, fl, L = costruisci(schema, solo_stab)
            r = s.metriche_ci(y, S, fl, args.n_boot, args.seed)
            P(f"\n== etichette da {schema} - {tag} ==  finestre {len(y)} (arco {int((y == 1).sum())}, pre-arco {int((y == 0).sum())})")
            P(f"{'Modello':<24}{'DR %':>26}{'FPR %':>26}")
            for k in s.MODELLI:
                P(f"{s.ETICHETTE[k]:<24}{s.fmt(r[k], 'DR'):>26}{s.fmt(r[k], 'FPR'):>26}")
                righe.append([tag, schema, "tutte", s.ETICHETTE[k], len(y), f"{r[k]['DR']:.3f}", f"{r[k]['DR_ci'][0]:.3f}",
                              f"{r[k]['DR_ci'][1]:.3f}", f"{r[k]['FPR']:.3f}", f"{r[k]['FPR_ci'][0]:.3f}", f"{r[k]['FPR_ci'][1]:.3f}"])
            riep[f"{schema} | {tag}"] = dict(n=int(len(y)), arco=int((y == 1).sum()), pre=int((y == 0).sum()),
                                             modelli={s.ETICHETTE[k]: r[k] for k in s.MODELLI})
            if schema == "tensione":
                P(f"\nDR per fascia di livello (finestre di arco secondo la tensione) - {tag}")
                P(f"{'Modello':<24}" + "".join(f"{f_:>30}" for f_ in FASCE))
                for fa in FASCE:
                    riep[f"tensione | {tag}"][f"n_arco_{fa}"] = int(((y == 1) & np.array([fascia(x) == fa for x in L])).sum())
                P(f"{'n finestre di arco':<24}" + "".join(f"{riep[f'tensione | {tag}'][f'n_arco_{fa}']:>30}" for fa in FASCE))
                for j, k in enumerate(s.MODELLI):
                    cel = []
                    for fa in FASCE:
                        m_ = (y == 1) & np.array([fascia(x) == fa for x in L])
                        p, lo, hi, n = boot_rate((S[m_, j] > 0), fl[m_], args.n_boot, args.seed)
                        cel.append(f"{p:.1f} [{lo:.1f}, {hi:.1f}]" if n else "n/d")
                        righe.append([tag, "tensione", fa, s.ETICHETTE[k], n, f"{p:.3f}", f"{lo:.3f}", f"{hi:.3f}", "", "", ""])
                        riep[f"tensione | {tag}"].setdefault("DR_per_fascia", {}).setdefault(s.ETICHETTE[k], {})[fa] = dict(DR=p, lo=lo, hi=hi, n=n)
                    P(f"{s.ETICHETTE[k]:<24}" + "".join(f"{c:>30}" for c in cel))
    with open(out / "valutazione_etichette.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file_set", "etichette", "fascia_livello", "modello", "n_finestre", "DR", "DR_lo", "DR_hi", "FPR", "FPR_lo", "FPR_hi"])
        w.writerows(righe)
    (out / "riepilogo_etichette_v3.json").write_text(json.dumps(riep, indent=1, ensure_ascii=False, default=float), encoding="utf-8")
    P("\nLimiti: i modelli sono addestrati con le etichette dalla corrente; t1_V e' una definizione euristica (soglie fissate a mano, "
      "riferimento nei primi 0.2 s); una sola esecuzione sul test set. Le fasce di livello contano finestre, non esperimenti.")
    (out / "report.txt").write_text("\n".join(s.LOG) + "\n", encoding="utf-8")
    P(f"\nFile scritti in: {out.resolve()}")


if __name__ == "__main__":
    main()
