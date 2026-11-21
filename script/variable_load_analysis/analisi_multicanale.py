#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analisi multicanale e fusione di segnali per il rilevamento di archi elettrici DC.

Valuta le combinazioni di canali fisici:
- Corrente (I)
- Tensione (V)
- Potenza istantanea (P = V * I)
su condizioni di carico statiche, stima causale e segnali di riferimento ideale.
"""
import argparse
import csv
import sys
from collections import deque
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[1] if (QUI.parents[1] / "dataset").is_dir() else QUI.parent
for p in [str(QUI), str(REPO / "script"), str(REPO / "script" / "casual_current_I_norm"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import norm_causale as nc
import test_set_analysis as a

CAN = ["I", "V", "P"]
MODI = ["statico", "causale", "riferimento"]
CLASSI = [("calo forte (B/A < 0.70)", 0.0, 0.70), ("calo lieve (0.70-0.95)", 0.70, 0.95),
          ("aumento lieve (1.05-1.50)", 1.05, 1.50), ("aumento forte (> 1.50)", 1.50, 99.0)]
INSIEMI = {"MS-RCFE + Ridge: solo I": ["I"], "MS-RCFE + Ridge: solo V": ["V"],
           "MS-RCFE + Ridge: I + V": ["I", "V"], "MS-RCFE + Ridge: I + V + P": ["I", "V", "P"]}


def rec_da_mat(path, ds):
    from scipy.io import loadmat
    m = loadmat(path, squeeze_me=False)
    k = {x.lower(): x for x in m if not x.startswith("__")}
    i = m[k["currentdata"]].flatten()[::ds].astype(np.float64)
    v = m[k["voltagedata"]].flatten()[::ds].astype(np.float64)
    n = min(len(i), len(v))
    i, v = i[:n], v[:n]
    pw = m[k["power"]].flatten()[::ds].astype(np.float64)[:n] if "power" in k else v * i
    return dict(I=i, V=v, P=pw)


def prepara(sig, p):
    t1, nom = p.find_t1(sig["I"])
    if t1 is None:
        return None
    nomc = {c: float(np.median(sig[c][:p.NOMINAL_N])) for c in CAN}
    nomc["I"] = float(nom)
    if any(abs(nomc[c]) < 1e-9 for c in CAN):
        return None
    return dict(sig=sig, t1=int(t1), nom=nomc)


def finestre_singole(rec, p, rng, cap):
    n = len(rec["sig"]["I"])
    st = np.arange(0, n - p.WIN_N + 1, p.STEP_N)
    pre = st[st + p.WIN_N <= rec["t1"]]
    arc = st[st >= rec["t1"]]
    sel = [(int(s), 0) for s in rng.permutation(pre)[:cap]] + [(int(s), 1) for s in rng.permutation(arc)[:cap]]
    if not sel:
        return None
    W = {c: np.stack([rec["sig"][c][s:s + p.WIN_N] / rec["nom"][c] for s, _ in sel]).astype(np.float32) for c in CAN}
    return W, np.array([l for _, l in sel])


def media_mobile(v, n):
    c = np.concatenate([[0.0], np.cumsum(v)])
    return (c[n:] - c[:-n]) / n


def innesco_tensione(rec, p, soglia_v):
    fs = p.FS_HZ
    V = np.asarray(rec["sig"]["V"], dtype=np.float64)
    if not np.all(np.isfinite(V)) or not np.all(np.isfinite(rec["sig"]["I"])):
        return None
    t1 = rec["t1"]
    if t1 < int(0.1 * fs):
        return None
    v = media_mobile(np.abs(V), int(0.01 * fs))
    v_pre = float(np.median(v[:int(0.1 * fs)]))
    a, b = min(len(v) - 1, t1 + int(0.1 * fs)), min(len(v), t1 + int(0.6 * fs))
    if b - a < int(0.05 * fs):
        return None
    v_arc = float(np.median(v[a:b]))
    if v_arc < soglia_v or v_arc < 3 * max(v_pre, 1e-6):
        return None
    meta = (v_pre + v_arc) / 2
    c = np.concatenate([[0], np.cumsum((v > meta).astype(int))])
    m = int(0.05 * fs)
    idx = np.where(c[m:] - c[:-m] == m)[0]
    return int(idx[0]) if len(idx) else None


def con_tv(recs, p, args):
    out = []
    for r in recs:
        tv = innesco_tensione(r, p, args.soglia_v)
        if tv is not None:
            out.append(dict(r, tv=tv, t1_orig=r["t1"], t1=tv))
    return out


def carica(repo, meta, mats, p, ds):
    with open(repo / "dataset/dataset_new" / meta, newline="", encoding="utf-8") as f:
        fns = sorted({m["filename"] for m in csv.DictReader(f)})
    recs, saltati = [], 0
    for fn in fns:
        if fn not in mats:
            saltati += 1
            continue
        try:
            r = prepara(rec_da_mat(mats[fn], ds), p)
        except (KeyError, Exception):
            r = None
        if r is None:
            saltati += 1
        else:
            recs.append(r)
    print(f"{meta}: {len(recs)} registrazioni caricate, {saltati} saltate")
    return recs


def sint_ft(rng, p, A, durata=3.5, t_arco=1.5):
    n = int(durata * p.FS_HZ)
    k = int(t_arco * p.FS_HZ)
    i = A * (1 + 0.002 * rng.standard_normal(n))
    i[k:] = A * 0.85 * (1 + 0.012 * rng.standard_normal(n - k))
    v = 1.5 * (1 + 0.05 * rng.standard_normal(n))
    v[k:] = 30 * (1 + 0.05 * rng.standard_normal(n - k))
    return dict(I=i, V=v, P=v * i)


def etichette_finestre(st, tv, p):
    return np.where(st >= tv, 1, np.where(st + p.WIN_N <= tv, 0, -1))


def nominale_con_gate(x, accetta, p, W_s):
    win_n, step_n, nom_n = p.WIN_N, p.STEP_N, p.NOMINAL_N
    K = max(1, int(round(W_s / p.STEP_S)))
    buf = deque(maxlen=K)
    out, seeded, start, j = [], False, 0, 0
    n = len(x)
    while start + win_n <= n:
        end = start + win_n
        if not seeded and end >= nom_n:
            seeded = True
            s = 0
            while s + win_n <= start:
                buf.append(float(np.median(x[s:s + win_n])))
                s += step_n
        if len(buf) == 0 or not seeded:
            nom = float(np.median(x[:end]))
        else:
            nom = float(np.median(buf))
        out.append((start, nom))
        if seeded and accetta[j]:
            buf.append(float(np.median(x[start:end])))
        start += step_n
        j += 1
    return out


def fusione(ra, rb, p, pad_s, W_s, gate_frac):
    if ra["nom"]["I"] < p.MIN_NOM_A:
        return None
    pad_n = int(round(pad_s * p.FS_HZ / p.STEP_N)) * p.STEP_N
    reps = int(np.ceil(pad_n / p.NOMINAL_N))
    sig = {c: np.concatenate([np.tile(ra["sig"][c][:p.NOMINAL_N], reps)[:pad_n], rb["sig"][c]]) for c in CAN}
    stI = nc.nominale_causale(sig["I"], W_s, "level", p, None, gate_frac)
    acc = [a for _, _, a in stI]
    stime = {"I": {s: i for s, i, _ in stI}}
    for c in ("V", "P"):
        stime[c] = {s: v for s, v in nominale_con_gate(sig[c], acc, p, W_s)}
    t1 = rb["t1"] + pad_n
    starts, grp, lab = [], [], []
    for s in range(0, len(sig["I"]) - p.WIN_N + 1, p.STEP_N):
        if s + p.WIN_N <= pad_n:
            g, l = "A", 0
        elif s >= pad_n:
            if s >= t1:
                g, l = "B", 1
            elif s + p.WIN_N <= t1:
                g, l = "B", 0
            else:
                continue
        else:
            continue
        starts.append(s); grp.append(g); lab.append(l)
    if not starts:
        return None
    starts, grp, lab = np.array(starts), np.array(grp), np.array(lab)
    lv = np.array([sig["I"][s:s + p.WIN_N].mean() / rb["nom"]["I"] for s in starts])
    return dict(sig=sig, stime=stime, starts=starts, grp=grp, lab=lab, attivo=(lab == 1) & (lv >= 0.75),
                nom_a=ra["nom"], nom_b=rb["nom"])


def finestre_fuse(d, idx, modo, p):
    out = {}
    for c in CAN:
        w = np.stack([d["sig"][c][s:s + p.WIN_N] for s in d["starts"][idx]])
        if modo == "statico":
            den = d["nom_a"][c]
        elif modo == "riferimento":
            den = d["nom_b"][c]
        else:
            den = np.array([d["stime"][c][s] for s in d["starts"][idx]])[:, None]
        out[c] = (w / den).astype(np.float32)
    return out


def unisci(lista):
    return {c: np.concatenate([w[c] for w in lista]) for c in CAN}


def campiona_coppia(recs, rng, dmin):
    for _ in range(1000):
        a, b = rng.choice(len(recs), 2, replace=False)
        if abs(recs[b]["nom"]["I"] - recs[a]["nom"]["I"]) >= dmin:
            return recs[a], recs[b]
    return None


def costruisci_train(recs, p, rng, n_coppie, cap_s, cap_c, args):
    Ws, ys = [], []
    for r in recs:
        o = finestre_singole(r, p, rng, cap_s)
        if o:
            Ws.append(o[0]); ys.append(o[1])
    n_singole = sum(len(y) for y in ys)
    n_f = 0
    for _ in range(n_coppie):
        pr = campiona_coppia(recs, rng, args.dmin_A)
        if pr is None:
            continue
        d = fusione(pr[0], pr[1], p, args.pad, args.W, args.gate_frac)
        if d is None:
            continue
        idx = []
        for g, l in (("A", 0), ("B", 0), ("B", 1)):
            m = np.where((d["grp"] == g) & (d["lab"] == l))[0]
            idx += list(rng.permutation(m)[:cap_c])
        idx = np.array(sorted(idx), dtype=int)
        if len(idx) == 0:
            continue
        for modo in ("statico", "causale"):
            Ws.append(finestre_fuse(d, idx, modo, p)); ys.append(d["lab"][idx]); n_f += len(idx)
    return unisci(Ws), np.concatenate(ys), n_singole, n_f


def tab(W):
    return np.column_stack([W[c].mean(1) for c in CAN] + [W[c].std(1) for c in CAN])


def bundle(W, feat_fn):
    return dict(W=W, T=tab(W), F={c: feat_fn(W[c]) for c in CAN})


def soglia_migliore(v, y):
    best = None
    for segno in (1, -1):
        for t in np.quantile(v, np.linspace(0.01, 0.99, 197)):
            pred = (segno * v > segno * t)
            ba = (pred[y == 1].mean() + (~pred[y == 0]).mean()) * 50
            if best is None or ba > best[0]:
                best = (ba, segno, t)
    return best[1], best[2]


def addestra(Btr, ytr, soglia, alpha, seed):
    P = {}
    P["soglia su I (0.9413 dal paper)"] = lambda B: B["T"][:, 0] < soglia
    seg, t = soglia_migliore(Btr["T"][:, 0], ytr)
    P["soglia su I (ritarata sul training attuale)"] = lambda B, seg=seg, t=t: seg * B["T"][:, 0] > seg * t
    t2 = DecisionTreeClassifier(max_depth=2, random_state=seed, class_weight="balanced").fit(Btr["T"][:, [0, 1]], ytr)
    P["regola a soglie su I e V (albero prof. 2)"] = lambda B: t2.predict(B["T"][:, [0, 1]]).astype(bool)
    t4 = DecisionTreeClassifier(max_depth=4, random_state=seed, class_weight="balanced").fit(Btr["T"], ytr)
    P["albero prof. 4 su medie e dev. std. di I, V, P"] = lambda B: t4.predict(B["T"]).astype(bool)
    for nome, cs in INSIEMI.items():
        Ftr = np.hstack([Btr["F"][c] for c in cs])
        sc = StandardScaler().fit(Ftr)
        clf = RidgeClassifier(alpha=alpha, class_weight="balanced").fit(sc.transform(Ftr), ytr)
        P[nome] = lambda B, cs=cs, sc=sc, clf=clf: clf.decision_function(sc.transform(np.hstack([B["F"][c] for c in cs]))) > 0
    return P


def valuta_singole(P, B, y):
    righe = []
    for k, f in P.items():
        pred = f(B)
        dr, fpr = 100 * pred[y == 1].mean(), 100 * pred[y == 0].mean()
        righe.append(dict(metodo=k, DR=dr, FPR=fpr, BA=(dr + 100 - fpr) / 2))
    return righe


def diagnostica(B, y, recs, p):
    out = []
    for i, c in enumerate(CAN):
        a = roc_auc_score(y, B["T"][:, i])
        out.append(f"  AUC della media di {c} normalizzata (arco vs pre-arco, test): {100 * max(a, 1 - a):.1f}%")
    rv = [np.mean(r["sig"]["V"][r["t1"]:]) / np.mean(r["sig"]["V"][:r["t1"]]) for r in recs if r["t1"] > 0]
    rp = [np.mean(r["sig"]["P"][r["t1"]:]) / np.mean(r["sig"]["P"][:r["t1"]]) for r in recs if r["t1"] > 0]
    cost = np.mean([np.std(r["sig"]["V"]) < 1e-3 * abs(r["nom"]["V"]) for r in recs])
    out.append(f"  V dopo t1 / V prima di t1 (mediana): {np.median(rv):.3f}   P: {np.median(rp):.3f}")
    out.append(f"  registrazioni con V quasi costante (dev. std. < 0.1% del nominale): {100 * cost:.1f}%")
    try:
        from scipy.stats import spearmanr
        rho = spearmanr([r["nom"]["V"] for r in recs], [r["nom"]["I"] for r in recs])[0]
        out.append(f"  Spearman tra V nominale e I nominale (tra registrazioni): {rho:.3f}")
    except Exception:
        pass
    return out


def valuta_fuse(P, recs_te, p, rng, args, feat_fn):
    acc = {}
    per_classe = {c[0]: 0 for c in CLASSI}
    quota = int(np.ceil(args.coppie / len(CLASSI)))
    tent = 0
    n_ok = 0
    while sum(per_classe.values()) < args.coppie and tent < 50 * args.coppie:
        tent += 1
        pr = campiona_coppia(recs_te, rng, args.dmin_A)
        if pr is None:
            break
        r = pr[1]["nom"]["I"] / pr[0]["nom"]["I"]
        cl = next((c[0] for c in CLASSI if c[1] <= r < c[2]), None)
        if cl is None or per_classe[cl] >= quota:
            continue
        d = fusione(pr[0], pr[1], p, args.pad, args.W, args.gate_frac)
        if d is None:
            continue
        per_classe[cl] += 1; n_ok += 1
        idx = np.arange(len(d["starts"]))
        for modo in MODI:
            B = bundle(finestre_fuse(d, idx, modo, p), feat_fn)
            m_pre = (d["grp"] == "B") & (d["lab"] == 0)
            m_arc = d["attivo"]
            m_A = (d["grp"] == "A") if modo != "riferimento" else np.zeros(len(idx), bool)
            for k, f in P.items():
                pred = f(B)
                a = acc.setdefault((cl, modo, k), dict(fp=0, npre=0, tp=0, narc=0, fa=0, nA=0))
                a["fp"] += int((pred & m_pre).sum()); a["npre"] += int(m_pre.sum())
                a["tp"] += int((pred & m_arc).sum()); a["narc"] += int(m_arc.sum())
                a["fa"] += int((pred & m_A).sum()); a["nA"] += int(m_A.sum())
    ordine = [c[0] for c in CLASSI]
    righe = []
    for (cl, modo, k), a in sorted(acc.items(), key=lambda x: (ordine.index(x[0][0]), MODI.index(x[0][1]), list(P).index(x[0][2]))):
        righe.append(dict(classe=cl, normalizzazione=modo, metodo=k,
                          FPR_B_pre=100 * a["fp"] / max(a["npre"], 1), DR_arco_attivo=100 * a["tp"] / max(a["narc"], 1),
                          FPR_regime_A=(100 * a["fa"] / a["nA"]) if a["nA"] else float("nan")))
    return righe, per_classe


def stampa_fuse(righe, solo=None):
    cl_prec = None
    for r in righe:
        if solo and r["normalizzazione"] not in solo:
            continue
        if r["classe"] != cl_prec:
            print(f"\n=== {r['classe']}")
            print(f"  {'normalizzazione':<14}{'metodo':<50}{'FPR carico cambia':>19}{'DR arco attivo':>16}")
            cl_prec = r["classe"]
        print(f"  {r['normalizzazione']:<14}{r['metodo']:<50}{r['FPR_B_pre']:>18.1f}%{r['DR_arco_attivo']:>15.1f}%")


def salva(righe, nome, out):
    Path(out).mkdir(parents=True, exist_ok=True)
    with open(Path(out) / nome, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
        w.writeheader(); w.writerows(righe)


def esegui(recs_tr, recs_te, p, args, feat_fn, rng):
    Wtr, ytr, ns, nf = costruisci_train(recs_tr, p, rng, args.coppie_train, args.cap_singole, args.cap_coppia, args)
    print(f"training: {len(ytr)} finestre ({ns} da registrazioni singole, {nf} da coppie fuse); arco {int(ytr.sum())}, pre-arco {int((1 - ytr).sum())}")
    Btr = bundle(Wtr, feat_fn)
    P = addestra(Btr, ytr, args.soglia, args.alpha, args.seed)
    Wte, yte = [], []
    for r in recs_te:
        o = finestre_singole(r, p, rng, 10 ** 9)
        if o:
            Wte.append(o[0]); yte.append(o[1])
    Wte, yte = unisci(Wte), np.concatenate(yte)
    Bte = bundle(Wte, feat_fn)
    print("\nDiagnostica dei canali (test set):")
    for s in diagnostica(Bte, yte, recs_te, p):
        print(s)
    rs = valuta_singole(P, Bte, yte)
    print(f"\n=== Test set a carico costante (finestre ricostruite dai .mat: {len(yte)})")
    print(f"  {'metodo':<50}{'DR':>8}{'FPR':>8}{'BA':>8}")
    for r in rs:
        print(f"  {r['metodo']:<50}{r['DR']:>8.1f}{r['FPR']:>8.1f}{r['BA']:>8.1f}")
    rf, pc = valuta_fuse(P, recs_te, p, rng, args, feat_fn)
    print(f"\ncoppie fuse di test: {pc}")
    return rs, rf


def sint(rng, p, A, scen, durata=3.0, t_arco=0.5, livello=0.88):
    n = int(durata * p.FS_HZ); k = int(t_arco * p.FS_HZ)
    i = A * (1 + 0.01 * rng.standard_normal(n))
    i[k:] = A * livello * (1 + 0.01 * rng.standard_normal(n - k))
    if scen == "informativa":
        v = 50 * (1 + 0.01 * rng.standard_normal(n))
        v[k:] = 50 * 1.15 * (1 + 0.01 * rng.standard_normal(n - k))
    else:
        v = 10 * i
    return dict(I=i, V=v, P=v * i)


def feat_prova(X):
    d = np.diff(X, axis=1)
    return np.column_stack([X.mean(1), X.std(1), d.std(1), X.min(1), X.max(1)]).astype(np.float64)


def selftest(p, args):
    ok = True
    rng0 = np.random.default_rng(0)
    sig = sint(rng0, p, 3.0, "informativa")
    r = prepara(sig, p)
    acc = [a for _, _, a in nc.nominale_causale(sig["I"], 2.0, "level", p, None, 0.98)]
    a = [i for s, i, _ in nc.nominale_causale(sig["I"], 2.0, "level", p, None, 0.98)]
    b = [v for _, v in nominale_con_gate(sig["I"], acc, p, 2.0)]
    ok &= np.allclose(a, b)
    print("stimatore con gate esterno = stimatore causale su I:", "OK" if np.allclose(a, b) else "FALLITO")
    for scen in ("informativa", "proporzionale"):
        rng = np.random.default_rng(1)
        livelli = [1.5, 2.0, 3.0, 4.0, 5.0]
        mk = lambda n: [prepara(sint(rng, p, float(rng.choice(livelli)), scen), p) for _ in range(n)]
        args.coppie, args.coppie_train = 24, 120
        print(f"\n########## selftest, tensione {scen}")
        rs, rf = esegui(mk(60), mk(30), p, args, feat_prova, rng)
        t = {(r["classe"], r["normalizzazione"], r["metodo"]): r for r in rf}
        cl = "calo forte (B/A < 0.70)"
        stampa_fuse([r for r in rf if r["classe"] in (cl, "aumento forte (> 1.50)")], solo=["causale"])
        solo_i = t[(cl, "causale", "MS-RCFE + Ridge: solo I")]
        iv = t[(cl, "causale", "MS-RCFE + Ridge: I + V")]
        if scen == "informativa":
            ok &= solo_i["FPR_B_pre"] + (100 - solo_i["DR_arco_attivo"]) > 60
            ok &= iv["FPR_B_pre"] < 15 and iv["DR_arco_attivo"] > 85
        else:
            ok &= abs(iv["FPR_B_pre"] - solo_i["FPR_B_pre"]) < 25
        print(f"selftest {scen}:", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Confronta soglia, regole a soglie e classificatori con uno o piu' canali (I, V, P) su coppie di registrazioni fuse con carico diverso")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_multicanale")
    ap.add_argument("--coppie", type=int, default=60)
    ap.add_argument("--coppie-train", type=int, default=300)
    ap.add_argument("--cap-singole", type=int, default=10)
    ap.add_argument("--cap-coppia", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dmin-A", type=float, default=0.5)
    ap.add_argument("--soglia", type=float, default=0.9413)
    ap.add_argument("--W", type=float, default=2.0)
    ap.add_argument("--gate-frac", type=float, default=0.98)
    ap.add_argument("--pad", type=float, default=4.0)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--downsample-factor", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.repo:
        repo = Path(args.repo)
    else:
        repo = Path.cwd()
        for cand in [QUI, QUI.parent, QUI.parent.parent, Path.cwd(), Path.cwd().parent, Path.cwd().parent.parent]:
            if (cand / "dataset").is_dir():
                repo = cand
                break
    repo = repo.resolve()
    p = nc.carica_parametri(repo)
    if args.selftest:
        sys.exit(selftest(p, args))
    if not args.mat_root:
        sys.exit("serve --mat-root (cartella dei .mat) oppure --selftest")

    import onnxruntime as ort
    sess = ort.InferenceSession(str(repo / "script/training/msrcfe/export_msrcfe/msrcfe.onnx"), providers=["CPUExecutionProvider"])
    nome_in = sess.get_inputs()[0].name

    def feat_fn(X):
        return np.vstack([sess.run(None, {nome_in: x.astype(np.float32).reshape(1, 1, -1)})[0].astype(np.float64) for x in X])

    mats = nc.indice_mat(args.mat_root)

    def carica(meta):
        with open(repo / "dataset/dataset_new" / meta, newline="", encoding="utf-8") as f:
            fns = sorted({m["filename"] for m in csv.DictReader(f)})
        recs, saltati = [], 0
        for fn in fns:
            if fn not in mats:
                saltati += 1
                continue
            try:
                r = prepara(rec_da_mat(mats[fn], args.downsample_factor), p)
            except KeyError:
                r = None
            if r is None:
                saltati += 1
            else:
                recs.append(r)
        print(f"{meta}: {len(recs)} registrazioni caricate, {saltati} saltate")
        return recs

    recs_tr = carica("arc_dataset_meta_train.csv")
    recs_te = carica("arc_dataset_meta_test.csv")
    if not recs_tr or not recs_te:
        sys.exit("nessuna registrazione con CurrentData e VoltageData trovata")
    rng = np.random.default_rng(args.seed)
    rs, rf = esegui(recs_tr, recs_te, p, args, feat_fn, rng)
    stampa_fuse(rf)
    salva(rs, "multicanale_singole.csv", args.out)
    salva(rf, "multicanale_fusione.csv", args.out)
    print(f"\nSalvato in {args.out}")


if __name__ == "__main__":
    main()
