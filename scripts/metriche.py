"""Regole condivise per etichetta modale, stabilità, accordo con la codifica umana ed errori di formato.

Definite una volta sola qui, così 04_aggregate.py, 05_metrics.py e gli script di analisi usano le stesse regole.
"""
import re
import warnings

import numpy as np
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
    """Etichetta solo con maggioranza stretta sulle run previste e nessuna run mancante."""
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
        zeri = np.zeros(len(etichette))
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


def fleiss_e_ta(conteggi, run_attese):
    """κ di Fleiss e TA@R (quota di risposte con tutte le run d'accordo).

    conteggi: una riga per risposta con tutte le run valide, una colonna per categoria.
    """
    if not len(conteggi) or run_attese <= 1:
        return float("nan"), float("nan")
    accordo_per_risposta = (conteggi.pow(2).sum(axis=1) - run_attese) / (run_attese * (run_attese - 1))
    proporzioni = conteggi.sum(axis=0) / (len(conteggi) * run_attese)
    atteso = (proporzioni**2).sum()
    kappa = (accordo_per_risposta.mean() - atteso) / (1 - atteso) if atteso < 1 else float("nan")
    return kappa, (conteggi.max(axis=1) == run_attese).mean()


# ---------------------------------------------------------------- errori di formato
# Fonti del dato, una per chiamata (colonna fonte_errori_formato del riepilogo):
FONTE_CAMPO = "campo errori_formato"
FONTE_MESSAGGI = "lista errori (messaggi classificati)"
FONTE_ASSENTE = "non disponibile"
# In "errori" 03_run.py scrive gli errori API come "NomeEccezione: messaggio" (es. "NotFoundError:
# Error code: 404"); quelli di formato vengono da valida() o dal rifiuto del modello e iniziano con
# una frase (es. "categoria non ammessa: ''"). Serve solo per le chiamate senza errori_formato.
_ERRORE_API = re.compile(r"^[A-Za-z_][\w.]*: ")


def errore_di_formato(messaggio):
    return not _ERRORE_API.match(str(messaggio))


def errori_formato_chiamata(record):
    """(errori di formato, fonte) di una chiamata; (None, FONTE_ASSENTE) se il dato manca."""
    campo = record.get("errori_formato")
    if campo is not None and not pd.isna(campo):
        return int(campo), FONTE_CAMPO
    errori = record.get("errori")
    if isinstance(errori, list):
        return sum(errore_di_formato(e) for e in errori), FONTE_MESSAGGI
    return None, FONTE_ASSENTE


def tasso_errori_formato(calls):
    """Tasso di errori di formato di un gruppo di chiamate (tutte, anche quelle poi ritentate).

    Denominatore: risposte ricevute dal modello = errori di formato + chiamate valide.
    Errori API e 429 non arrivano al modello e restano fuori da entrambi i termini.
    """
    n_errori = n_risposte = 0
    fonti = set()
    for record in calls.to_dict("records"):
        n, fonte = errori_formato_chiamata(record)
        fonti.add(fonte)
        if n is None:
            continue
        n_errori += n
        n_risposte += n + bool(record.get("valido"))
    return {
        "n_errori_formato": n_errori,
        "n_risposte_modello": n_risposte,
        "tasso_errori_formato": n_errori / n_risposte if n_risposte else float("nan"),
        "fonte_errori_formato": " + ".join(sorted(fonti)) if fonti else FONTE_ASSENTE,
    }
