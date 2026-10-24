"""Regole condivise per etichetta modale, gruppi di stabilità e accordo con la codifica umana.

Le usano 04_aggregate.py, 05_metrics.py e report_serbatoio.py, così le regole sono le stesse ovunque.
"""
import warnings

import pandas as pd
from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

from common import GRUPPI_STABILITA, INSTABILE

# Etichetta data alle risposte senza etichetta modale nella versione "controllo": non coincide mai
# con quella umana e non è una categoria, quindi nel F1 abbassa solo il richiamo della categoria umana.
SENZA_ETICHETTA = "(senza etichetta modale)"

VERSIONI = {
    "principale": "solo risposte con etichetta modale (con copertura)",
    "controllo": "tutte le risposte, quelle senza etichetta modale contate come errore",
    "confronto_modelli": "solo risposte con etichetta modale per tutti i modelli",
}


def etichetta_modale(categorie_valide, n_run_attese):
    """Regole M4: etichetta solo con maggioranza stretta sulle run previste e nessuna run mancante."""
    conteggi = pd.Series(categorie_valide, dtype=object).value_counts()
    massimo = int(conteggi.iloc[0]) if len(conteggi) else 0
    pareggio = len(conteggi) > 1 and conteggi.iloc[0] == conteggi.iloc[1]
    maggioranza = massimo > n_run_attese / 2
    run_mancanti = len(categorie_valide) < n_run_attese
    return {
        "etichetta_modale": conteggi.index[0] if maggioranza and not run_mancanti else None,
        "quota_modale": massimo / n_run_attese if n_run_attese else 0.0,
        "pareggio": pareggio,
        "da_revisionare": not maggioranza or run_mancanti,
        "run_mancanti": run_mancanti,
        "conteggi": conteggi,
    }


def gruppo_stabilita(etichetta, quota_modale):
    """Gruppo di common.GRUPPI_STABILITA; senza etichetta modale si va sempre in "instabile"."""
    if etichetta is None or pd.isna(etichetta):
        return INSTABILE
    for nome, minimo in sorted(GRUPPI_STABILITA.items(), key=lambda kv: -kv[1]):
        if quota_modale >= minimo - 1e-9:
            return nome
    return INSTABILE


def id_con_etichetta_per_tutti(agg):
    """{(esperimento_id, strategia): id con etichetta modale per tutti i modelli di quell'esperimento}."""
    risultato = {}
    for chiave, g in agg.groupby(["esperimento_id", "strategia"]):
        con_etichetta = (
            g.assign(ok=g["etichetta_modale"].notna())
            .pivot_table(index="id", columns="modello", values="ok", aggfunc="all")
            .reindex(columns=g["modello"].unique())
            .fillna(False)
            .astype(bool)
        )
        risultato[chiave] = set(con_etichetta.index[con_etichetta.all(axis=1)])
    return risultato


def accordo(g, etichette, versione, id_comuni=None):
    """κ di Cohen (con po e pe), accordo percentuale e precision/recall/F1 per categoria
    in una delle tre VERSIONI."""
    g = g.dropna(subset=["umano"])
    n_totale = len(g)
    if versione == "principale":
        g = g.dropna(subset=["etichetta_modale"])
    elif versione == "confronto_modelli":
        g = g[g["id"].isin(id_comuni or set())]
    elif versione != "controllo":
        raise ValueError(f"versione sconosciuta: {versione}")
    y_umano = g["umano"].astype(str)
    y_llm = g["etichetta_modale"].fillna(SENZA_ETICHETTA).astype(str)
    risultato = {
        "n": len(g),
        "n_totale": n_totale,
        "copertura": len(g) / n_totale if n_totale else float("nan"),
        "y_umano": y_umano,
        "y_llm": y_llm,
    }
    if not len(g):
        zeri = pd.Series([0.0] * len(etichette)).to_numpy()
        return risultato | {"kappa": float("nan"), "po": float("nan"), "pe": float("nan"),
                            "accordo_percentuale": float("nan"), "precision": zeri,
                            "recall": zeri, "f1": zeri, "supporto": zeri.astype(int),
                            "f1_macro": float("nan")}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # κ indefinito (NaN) se l'insieme ha una sola categoria
        kappa = cohen_kappa_score(y_umano, y_llm, labels=etichette + [SENZA_ETICHETTA])
    precisioni, richiami, f1, supporto = precision_recall_fscore_support(
        y_umano, y_llm, labels=etichette, zero_division=0
    )
    presenti = supporto > 0
    # κ = (po - pe) / (1 - pe): po accordo osservato, pe accordo atteso dai marginali
    po = (y_umano == y_llm).mean()
    pe = sum(
        (y_umano == e).mean() * (y_llm == e).mean() for e in etichette + [SENZA_ETICHETTA]
    )
    return risultato | {
        "kappa": kappa,
        "po": po,
        "pe": pe,
        "accordo_percentuale": po,
        "precision": precisioni,
        "recall": richiami,
        "f1": f1,
        "supporto": supporto,
        # F1 macro solo sulle categorie con almeno un caso umano nell'insieme considerato
        "f1_macro": f1[presenti].mean() if presenti.any() else float("nan"),
    }
