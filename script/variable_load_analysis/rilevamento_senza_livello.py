#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rilevamento dell'arco elettrico DC indipendente dal livello di corrente continua.

Estrae feature spettrali (densita' spettrale di potenza FFT suddivisa in 5 bande)
e descrittori di forma privi di offset DC, dimostrando che la classificazione
dell'arco si basa sull'informazione dinamica ad alta frequenza e non sulla sola
caduta di livello.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler

QUI = Path(__file__).resolve().parent
REPO = QUI.parents[1] if (QUI.parents[1] / "dataset").is_dir() else QUI.parent
for p in [str(QUI), str(REPO / "script"), str(REPO / "script" / "casual_current_I_norm"), str(REPO)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import analisi_multicanale as tm

nc = tm.nc

FASCE = [(10, 100), (100, 500), (500, 1000), (1000, 2000), (2000, 5000)]


def energia_bande(X, fs=10000):
    X = X - X.mean(1, keepdims=True)
    S = np.abs(np.fft.rfft(X * np.hanning(X.shape[1]), axis=1)) ** 2
    f = np.fft.rfftfreq(X.shape[1], 1 / fs)
    tot = S[:, 1:].sum(1) + 1e-12
    return np.column_stack([np.log(S[:, (f >= a) & (f < b)].sum(1) / tot + 1e-12) for a, b in FASCE])



def media(X):
    return X.mean(1, keepdims=True)


def t_forma(X):
    return (X - media(X)) / (X.std(1, keepdims=True) + 1e-9)


def feat_forma(X):
    m = media(X)
    r = X / np.where(np.abs(m) < 1e-9, 1e-9, m)
    d = np.diff(r, axis=1)
    return np.column_stack([r.std(1), d.std(1), np.abs(d).max(1), r.max(1) - r.min(1)])


def scalari(X):
    return np.column_stack([X.mean(1), feat_forma(X)])


class Blocco:
    def __init__(self, X, feat_fn):
        self.X = X
        self.f = feat_fn
        self.c = {}

    def F(self, nome):
        if nome not in self.c:
            self.c[nome] = self.f(self.X) if nome == "liv" else (self.f(self.X - media(self.X)) if nome == "ac" else self.f(t_forma(self.X)))
        return self.c[nome]


def costruisci(Btr, y, alpha, soglia):
    X = Btr.X
    P = {}
    P["soglia sul livello (0.9413 del paper)"] = lambda B: B.X.mean(1) < soglia
    seg, t = tm.soglia_migliore(X.mean(1), y)
    P["soglia sul livello (ritarata sul training)"] = lambda B, seg=seg, t=t: seg * B.X.mean(1) > seg * t
    ftr = feat_forma(X)
    best = None
    for i in range(ftr.shape[1]):
        sg, tt = tm.soglia_migliore(ftr[:, i], y)
        pr = sg * ftr[:, i] > sg * tt
        ba = (pr[y == 1].mean() + (~pr[y == 0]).mean()) * 50
        if best is None or ba > best[0]:
            best = (ba, i, sg, tt)
    _, i, sg, tt = best
    P["soglia sulla migliore caratteristica di forma"] = lambda B, i=i, sg=sg, tt=tt: sg * feat_forma(B.X)[:, i] > sg * tt
    sc = StandardScaler().fit(ftr)
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(ftr), y)
    P["logistica su 4 caratteristiche di forma"] = lambda B, sc=sc, lr=lr: lr.decision_function(sc.transform(feat_forma(B.X))) > 0
    btr = energia_bande(X)
    best_b = None
    for i in range(btr.shape[1]):
        sg, tt = tm.soglia_migliore(btr[:, i], y)
        pr = sg * btr[:, i] > sg * tt
        ba = (pr[y == 1].mean() + (~pr[y == 0]).mean()) * 50
        if best_b is None or ba > best_b[0]:
            best_b = (ba, i, sg, tt)
    _, ib, sgb, ttb = best_b
    P["soglia sulla migliore banda spettrale"] = lambda B, ib=ib, sgb=sgb, ttb=ttb: sgb * energia_bande(B.X)[:, ib] > sgb * ttb
    scb = StandardScaler().fit(btr)
    lrb = LogisticRegression(max_iter=2000, class_weight="balanced").fit(scb.transform(btr), y)
    P["logistica sulle bande spettrali"] = lambda B, scb=scb, lrb=lrb: lrb.decision_function(scb.transform(energia_bande(B.X))) > 0
    varianti = {
        "MS-RCFE + Ridge: livello": lambda B: B.F("liv"),
        "MS-RCFE + Ridge: solo forma": lambda B: B.F("forma"),
        "MS-RCFE + Ridge: media rimossa (ampiezza AC tenuta)": lambda B: B.F("ac"),
        "MS-RCFE + Ridge: forma + livello e 4 scalari": lambda B: np.hstack([B.F("forma"), scalari(B.X)]),
        "MS-RCFE + Ridge: forma + livello + bande spettrali": lambda B: np.hstack([B.F("forma"), scalari(B.X), energia_bande(B.X)]),
        "Ridge: livello + bande spettrali (senza MS-RCFE)": lambda B: np.hstack([media(B.X), energia_bande(B.X)]),
    }
    for nome, fn in varianti.items():
        F = fn(Btr)
        sc2 = StandardScaler().fit(F)
        clf = RidgeClassifier(alpha=alpha, class_weight="balanced").fit(sc2.transform(F), y)
        P[nome] = lambda B, fn=fn, sc2=sc2, clf=clf: clf.decision_function(sc2.transform(fn(B))) > 0
    return P


def prepara_coppie(recs_te, p, rng, args):
    out = []
    per = {c[0]: 0 for c in tm.CLASSI}
    quota = int(np.ceil(args.coppie / len(tm.CLASSI)))
    tent = 0
    while sum(per.values()) < args.coppie and tent < 50 * args.coppie:
        tent += 1
        pr = tm.campiona_coppia(recs_te, rng, args.dmin_A)
        if pr is None:
            break
        r = pr[1]["nom"]["I"] / pr[0]["nom"]["I"]
        cl = next((c[0] for c in tm.CLASSI if c[1] <= r < c[2]), None)
        if cl is None or per[cl] >= quota:
            continue
        d = tm.fusione(pr[0], pr[1], p, args.pad, args.W, args.gate_frac)
        if d is None:
            continue
        per[cl] += 1
        idx = np.arange(len(d["starts"]))
        out.append(dict(classe=cl, grp=d["grp"], lab=d["lab"], attivo=d["attivo"],
                        X={m: tm.finestre_fuse(d, idx, m, p)["I"] for m in tm.MODI}))
    return out, per


def valuta_fuse(P, coppie, feat_fn):
    acc = {}
    for d in coppie:
        m_pre = (d["grp"] == "B") & (d["lab"] == 0)
        m_arc = d["attivo"]
        for modo in tm.MODI:
            B = Blocco(d["X"][modo], feat_fn)
            for k, f in P.items():
                pred = f(B)
                a = acc.setdefault((d["classe"], modo, k), dict(fp=0, npre=0, tp=0, narc=0))
                a["fp"] += int((pred & m_pre).sum()); a["npre"] += int(m_pre.sum())
                a["tp"] += int((pred & m_arc).sum()); a["narc"] += int(m_arc.sum())
    return {k: (100 * a["fp"] / max(a["npre"], 1), 100 * a["tp"] / max(a["narc"], 1)) for k, a in acc.items()}


def stampa_fuse(nome_regime, P, ris, righe_csv):
    print(f"\n########## training: {nome_regime}")
    print("  celle: FPR / DR in %  (FPR = falsi allarmi dopo il cambio di carico, DR = arco attivo dopo il cambio)")
    for cl in [c[0] for c in tm.CLASSI]:
        print(f"\n=== {cl}")
        print(f"  {'metodo':<48}{'statico':>16}{'causale':>16}{'riferimento':>16}")
        for k in P:
            celle = []
            for modo in tm.MODI:
                v = ris.get((cl, modo, k))
                celle.append("-" if v is None else f"{v[0]:5.1f} / {v[1]:5.1f}")
                if v is not None:
                    righe_csv.append(dict(training=nome_regime, classe=cl, normalizzazione=modo, metodo=k, FPR=v[0], DR=v[1]))
            print(f"  {k:<48}{celle[0]:>16}{celle[1]:>16}{celle[2]:>16}")


def valuta_singole(P, B, y):
    righe = []
    for k, f in P.items():
        pred = f(B)
        dr, fpr = 100 * pred[y == 1].mean(), 100 * pred[y == 0].mean()
        righe.append((k, dr, fpr, (dr + 100 - fpr) / 2))
    return righe


def esegui(recs_tr, recs_te, p, args, feat_fn, rng):
    Wte, yte = [], []
    for r in recs_te:
        o = tm.finestre_singole(r, p, rng, 10 ** 9)
        if o:
            Wte.append(o[0]); yte.append(o[1])
    Wte, yte = tm.unisci(Wte), np.concatenate(yte)
    Bte = Blocco(Wte["I"], feat_fn)
    coppie, per = prepara_coppie(recs_te, p, rng, args)
    print(f"coppie fuse di test: {per}")
    regimi = [("solo registrazioni a carico costante (come nel paper)", 0),
              ("registrazioni a carico costante + coppie fuse", args.coppie_train)]
    righe_csv, righe_s = [], []
    tutti = {}
    for nome, n_coppie in regimi:
        Wtr, ytr, ns, nf = tm.costruisci_train(recs_tr, p, rng, n_coppie, args.cap_singole, args.cap_coppia, args)
        print(f"\ntraining '{nome}': {len(ytr)} finestre ({ns} singole, {nf} da coppie fuse)")
        P = costruisci(Blocco(Wtr["I"], feat_fn), ytr, args.alpha, args.soglia)
        print(f"\n=== Test set a carico costante, training '{nome}' ({len(yte)} finestre)")
        print(f"  {'metodo':<48}{'DR':>8}{'FPR':>8}{'BA':>8}")
        for k, dr, fpr, ba in valuta_singole(P, Bte, yte):
            print(f"  {k:<48}{dr:>8.1f}{fpr:>8.1f}{ba:>8.1f}")
            righe_s.append(dict(training=nome, metodo=k, DR=dr, FPR=fpr, BA=ba))
        ris = valuta_fuse(P, coppie, feat_fn)
        stampa_fuse(nome, P, ris, righe_csv)
        tutti[nome] = ris
    return righe_s, righe_csv, tutti


def sint2(rng, p, A, durata=3.0, t_arco=0.5, livello=0.88):
    n = int(durata * p.FS_HZ); k = int(t_arco * p.FS_HZ)
    t = np.arange(n) / p.FS_HZ
    rip = 0.01 * np.sin(2 * np.pi * 120 * t + rng.uniform(0, 6.28))
    i = A * (1 + 0.002 * rng.standard_normal(n) + rip)
    pic = np.clip(0.006 * rng.standard_t(3, size=n - k), -0.05, 0.05)
    i[k:] = A * livello * (1 + pic)
    v = 50 * (1 + 0.003 * rng.standard_normal(n))
    return dict(I=i, V=v, P=v * i)


def feat_prova(X):
    d = np.diff(X, axis=1)
    return np.round(np.column_stack([X.mean(1), X.std(1), d.std(1), np.abs(d).max(1), X.min(1), X.max(1)]), 5).astype(np.float64)


def selftest(p, args):
    rng = np.random.default_rng(1)
    livelli = [1.5, 2.0, 3.0, 4.0, 5.0]
    mk = lambda n: [tm.prepara(sint2(rng, p, float(rng.choice(livelli))), p) for _ in range(n)]
    args.coppie, args.coppie_train = 24, 120
    rs, rc, tutti = esegui(mk(60), mk(30), p, args, feat_prova, rng)
    nome = "registrazioni a carico costante + coppie fuse"
    t = tutti[nome]
    cl = "calo forte (B/A < 0.70)"
    forma = t[(cl, "causale", "MS-RCFE + Ridge: solo forma")]
    soglia = t[(cl, "causale", "soglia sul livello (0.9413 del paper)")]
    cl2 = "aumento forte (> 1.50)"
    forma2 = t[(cl2, "causale", "MS-RCFE + Ridge: solo forma")]
    soglia2 = t[(cl2, "causale", "soglia sul livello (0.9413 del paper)")]
    banda = t[(cl, "causale", "logistica sulle bande spettrali")]
    ok = banda[0] < 20 and banda[1] > 80 and forma[0] < 20 and forma[1] > 80 and soglia[0] > 90 and forma2[0] < 20 and forma2[1] > 80 and soglia2[1] < 10
    print("\nselftest:", "OK" if ok else "FALLITO")
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="Classificatore senza livello e baseline su coppie di registrazioni fuse con carico diverso")
    ap.add_argument("--repo", default=None)
    ap.add_argument("--mat-root", default=None)
    ap.add_argument("--out", default="ris_senza_livello_fusione_v4")
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
                r = tm.prepara(tm.rec_da_mat(mats[fn], args.downsample_factor), p)
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
    rs, rc, _ = esegui(recs_tr, recs_te, p, args, feat_fn, rng)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for nome, righe in (("senza_livello_fusione_v4_singole.csv", rs), ("senza_livello_fusione_v4_coppie.csv", rc)):
        with open(Path(args.out) / nome, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(righe[0].keys()))
            w.writeheader(); w.writerows(righe)
    print(f"\nSalvato in {args.out}")


if __name__ == "__main__":
    main()
