"""Quali risposte mandare a revisione umana: regole a confronto sull'esperimento attivo (nessuna chiamata API).

Versione di controllo: una risposta senza etichetta modale conta come disaccordo con l'umano.
Gruppi di stabilità da common.GRUPPI_STABILITA. Richiede due modelli in config.yaml.
  1. sovrapposizione delle risposte instabili tra i modelli
  2. cosa fa l'altro modello quando uno è instabile e in disaccordo
  3. regole di revisione a-f, con i disaccordi calcolati rispetto a ciascun modello
Output: results/<domanda>/metrics/revisione_<esperimento_id>.csv (colonna "tabella").
"""
import numpy as np
import pandas as pd

from common import ABBASTANZA_STABILE, INSTABILE, METRICS, RESULTS, STABILE, carica_config, nome_file
from metriche import gruppo_stabilita

N_SIMULAZIONI = 10_000  # estrazioni per la revisione casuale (regola f)

cfg = carica_config()
esperimento_id = cfg["esperimento_id"]
strategia = cfg["strategie"][0]  # una sola strategia per esperimento
modelli = [m["nome"] for m in cfg["modelli"]]
if len(modelli) != 2:
    raise SystemExit("L'analisi richiede esattamente due modelli in config.yaml")

agg = pd.read_csv(RESULTS / "aggregated.csv", encoding="utf-8-sig")
agg = agg[(agg["esperimento_id"] == esperimento_id) & (agg["strategia"] == strategia)]
agg["gruppo_stabilita"] = [
    gruppo_stabilita(e, q) for e, q in zip(agg["etichetta_modale"], agg["quota_modale"])
]

# Una riga per risposta, colonne per modello
umano = agg.groupby("id")["umano"].first()
etichetta = agg.pivot(index="id", columns="modello", values="etichetta_modale")[modelli]
gruppo = agg.pivot(index="id", columns="modello", values="gruppo_stabilita")[modelli]
ids = umano.index
if etichetta.isna().all(axis=None) or len(etichetta) != len(umano) or gruppo.isna().any(axis=None):
    raise SystemExit("Ogni risposta deve avere una riga aggregata per entrambi i modelli")
N = len(ids)
senza = etichetta.isna()
disaccordo = senza | etichetta.ne(umano, axis=0)  # versione di controllo
instabile = gruppo.eq(INSTABILE)

righe = []


def pct(x, totale):
    return round(100 * x / totale, 1) if totale else None


# ---------------------------------------------------------------- 1. sovrapposizione
a, b = modelli
for sottoinsieme, maschera in (
    ("instabili", instabile),
    ("instabili in disaccordo con l'umano", instabile & disaccordo),
):
    for voce, sel in (
        (f"solo {a}", maschera[a] & ~maschera[b]),
        (f"solo {b}", maschera[b] & ~maschera[a]),
        ("entrambi", maschera[a] & maschera[b]),
        (f"{a} (totale)", maschera[a]),
        (f"{b} (totale)", maschera[b]),
        ("almeno uno", maschera[a] | maschera[b]),
    ):
        righe.append({"tabella": "1_sovrapposizione", "sottoinsieme": sottoinsieme, "voce": voce,
                      "n": int(sel.sum()), "percentuale": pct(sel.sum(), N)})

# ---------------------------------------------------------------- 2. secondo parere
for modello, altro in ((a, b), (b, a)):
    casi = instabile[modello] & disaccordo[modello]
    n_casi = int(casi.sum())
    stato = pd.Series(index=ids, dtype=object)
    stato[~disaccordo[altro]] = "d'accordo con l'umano"
    stato[disaccordo[altro] & gruppo[altro].eq(STABILE)] = "stabile in disaccordo"
    stato[disaccordo[altro] & gruppo[altro].eq(ABBASTANZA_STABILE)] = "abbastanza stabile in disaccordo"
    stato[disaccordo[altro] & gruppo[altro].eq(INSTABILE) & ~senza[altro]] = "instabile (in disaccordo)"
    stato[senza[altro]] = "senza etichetta"
    stessa = etichetta[altro].eq(etichetta[modello]) & ~senza[modello]
    for voce in ("d'accordo con l'umano", "stabile in disaccordo", "abbastanza stabile in disaccordo",
                 "instabile (in disaccordo)", "senza etichetta"):
        sel = casi & stato.eq(voce)
        riga = {"tabella": "2_secondo_parere", "modello_instabile": modello, "altro_modello": altro,
                "voce": voce, "n": int(sel.sum()), "percentuale": pct(sel.sum(), n_casi),
                "base": n_casi}
        if "disaccordo" in voce:
            riga["stessa_etichetta_del_modello_instabile"] = int((sel & stessa).sum())
            riga["etichetta_diversa"] = int((sel & ~stessa).sum())
        righe.append(riga)

# ---------------------------------------------------------------- 3. regole di revisione
regola_c = instabile[a] | instabile[b]
regola_d = senza[a] | senza[b] | etichetta[a].ne(etichetta[b])
regola_e = regola_c | regola_d
rng = np.random.default_rng(cfg["seed"])
for riferimento in modelli:
    dis = disaccordo[riferimento]
    D = int(dis.sum())
    regole = {
        "a. instabile per il modello stesso": instabile[riferimento],
        "b. instabile o abbastanza stabile per il modello stesso": gruppo[riferimento].ne(STABILE),
        "c. instabile per almeno uno dei due modelli": regola_c,
        "d. etichette modali diverse (o una mancante)": regola_d,
        "e. c oppure d (regola del passo 6)": regola_e,
    }
    for nome, sel in regole.items():
        n_rivedute, intercettati = int(sel.sum()), int((sel & dis).sum())
        righe.append({
            "tabella": "3_regole_revisione", "riferimento": riferimento, "voce": nome,
            "n": n_rivedute, "percentuale": pct(n_rivedute, N),
            "disaccordi_totali": D, "disaccordi_intercettati": intercettati,
            "percentuale_disaccordi_intercettati": pct(intercettati, D),
            "tasso_disaccordo_tra_rivedute": pct(intercettati, n_rivedute),
        })
    # f. stessa quantità di e, scelta a caso: media e intervallo al 95% su N_SIMULAZIONI estrazioni
    n_e = int(regola_e.sum())
    valori = dis.to_numpy()
    estratti = np.array([valori[rng.choice(N, n_e, replace=False)].sum() for _ in range(N_SIMULAZIONI)])
    basso, alto = np.percentile(estratti, [2.5, 97.5])
    righe.append({
        "tabella": "3_regole_revisione", "riferimento": riferimento,
        "voce": f"f. casuale, stessa quantità di e (media {N_SIMULAZIONI:_} estrazioni)".replace("_", "."),
        "n": n_e, "percentuale": pct(n_e, N), "disaccordi_totali": D,
        "disaccordi_intercettati": round(float(estratti.mean()), 1),
        "percentuale_disaccordi_intercettati": pct(estratti.mean(), D),
        "tasso_disaccordo_tra_rivedute": pct(estratti.mean(), n_e),
        "intercettati_ic95": f"{basso:.0f}-{alto:.0f}",
    })

tabelle = pd.DataFrame(righe)
tabelle.insert(0, "esperimento_id", esperimento_id)
percorso = METRICS / f"revisione_{nome_file(esperimento_id)}.csv"
METRICS.mkdir(parents=True, exist_ok=True)
tabelle.to_csv(percorso, index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- riepilogo
print(f"Revisione umana | {esperimento_id} | valutazione, {N} risposte | versione di controllo")
print("\n1. Risposte instabili per modello")
for sottoinsieme, t in tabelle[tabelle["tabella"] == "1_sovrapposizione"].groupby("sottoinsieme", sort=False):
    print(f"  {sottoinsieme}:")
    for r in t.itertuples():
        print(f"    {r.voce:<28} {r.n:>4}  ({r.percentuale}%)")
print("\n2. Secondo parere sulle risposte instabili e in disaccordo")
for (modello, altro), t in tabelle[tabelle["tabella"] == "2_secondo_parere"].groupby(
    ["modello_instabile", "altro_modello"], sort=False
):
    print(f"  {modello} instabile e in disaccordo ({int(t['base'].iloc[0])} risposte): cosa fa {altro}")
    for r in t.itertuples():
        extra = (f"  [stessa etichetta: {int(r.stessa_etichetta_del_modello_instabile)}, "
                 f"diversa: {int(r.etichetta_diversa)}]" if "disaccordo" in r.voce else "")
        print(f"    {r.voce:<34} {r.n:>4}  ({r.percentuale}%){extra}")
print("\n3. Regole di revisione (da rivedere | disaccordi intercettati | tasso di disaccordo tra le rivedute)")
for riferimento, t in tabelle[tabelle["tabella"] == "3_regole_revisione"].groupby("riferimento", sort=False):
    print(f"  Disaccordi rispetto a {riferimento}: {int(t['disaccordi_totali'].iloc[0])} su {N}")
    for r in t.itertuples():
        ic = f" (IC95 {r.intercettati_ic95})" if isinstance(r.intercettati_ic95, str) else ""
        intercettati = f"{r.disaccordi_intercettati:g}"
        print(f"    {r.voce:<60} {r.n:>4} ({r.percentuale}%) | "
              f"{intercettati:>4}{ic} ({r.percentuale_disaccordi_intercettati}%) | "
              f"{r.tasso_disaccordo_tra_rivedute}%")
print(f"\nScritto {percorso}")
