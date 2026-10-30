"""Percorsi, configurazione e funzioni condivise da tutti gli script."""
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import yaml
from ftfy import fix_text
from rapidfuzz import fuzz

# Su Windows la console non è UTF-8: senza questo accenti e "κ" danno errore
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"

# Chiave che identifica una singola chiamata: serve per riprendere dopo un'interruzione.
CHIAVE = ["esperimento_id", "set", "provider", "modello", "strategia", "run", "id"]

# Gruppi di stabilità sulla quota modale (run con l'etichetta modale / run previste):
#   stabile 10/10, abbastanza_stabile 8-9/10, instabile 7/10 o meno o senza etichetta modale.
# Valore = quota minima del gruppo.
STABILE, ABBASTANZA_STABILE, INSTABILE = "stabile", "abbastanza_stabile", "instabile"
GRUPPI_STABILITA = {STABILE: 1.0, ABBASTANZA_STABILE: 0.8, INSTABILE: 0.0}


def carica_config():
    """Legge config.yaml dalla cartella del progetto."""
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Split e risultati sono separati per domanda: cambiando domanda_attiva non si mescolano.
DOMANDA = carica_config()["dati"]["domanda_attiva"]
SPLITS = ROOT / "data" / "splits" / DOMANDA
RESULTS = ROOT / "results" / DOMANDA
CALLS = RESULTS / "calls.jsonl"
METRICS = RESULTS / "metrics"


def tutte_le_etichette(cfg):
    """Categorie del codebook più l'etichetta "non classificabile"."""
    return list(cfg["categorie"]) + [cfg["non_classificabile"]]


def leggi_split(nome):
    """Uno split della domanda attiva (serbatoio, valutazione o resto) come DataFrame."""
    return pd.read_csv(SPLITS / f"{nome}.csv", dtype={"id": str}, encoding="utf-8-sig")


def leggi_calls(percorso=None):
    """I record di calls.jsonl (lista vuota se il file non esiste).

    Una riga incompleta (processo ucciso mentre scriveva) si salta con un avviso:
    quella chiamata risulta non fatta e 03_run.py la rifà al rilancio.
    """
    percorso = percorso or CALLS
    if not percorso.exists():
        return []
    records = []
    with open(percorso, encoding="utf-8") as f:
        for n, riga in enumerate(f, 1):
            if not riga.strip():
                continue
            try:
                records.append(json.loads(riga))
            except json.JSONDecodeError:
                print(f"Avviso: riga {n} di {percorso} incompleta, ignorata", file=sys.stderr)
    return records


def tabella_calls():
    """calls.jsonl come DataFrame. I primi log non avevano esperimento_id e set:
    valgono come esperimento "legacy" sul campione di valutazione."""
    calls = pd.DataFrame(leggi_calls())
    if not calls.empty:
        for colonna, default in (("esperimento_id", "legacy"), ("set", "valutazione")):
            calls[colonna] = calls[colonna].fillna(default) if colonna in calls else default
    return calls


def normalizza(testo):
    """Minuscole, spazi compattati, apostrofi e virgolette uniformati: per confrontare le citazioni."""
    testo = unicodedata.normalize("NFKC", fix_text(str(testo))).lower()
    testo = testo.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    testo = testo.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
    return re.sub(r"\s+", " ", testo).strip()


def tratto_originale(testo, testo_normalizzato, inizio, fine):
    """Il tratto del testo originale che corrisponde a [inizio, fine) del testo normalizzato."""
    if inizio >= fine or not testo_normalizzato:
        return ""
    posizioni = [None] * len(testo_normalizzato)
    matcher = SequenceMatcher(None, testo, testo_normalizzato, autojunk=False)
    for operazione, raw_inizio, raw_fine, norm_inizio, norm_fine in matcher.get_opcodes():
        if operazione == "equal":
            for i in range(norm_inizio, norm_fine):
                posizioni[i] = raw_inizio + i - norm_inizio
        elif operazione in ("replace", "insert") and norm_fine > norm_inizio:
            lunghezza_raw = raw_fine - raw_inizio
            lunghezza_norm = norm_fine - norm_inizio
            for i in range(norm_inizio, norm_fine):
                if lunghezza_raw:
                    offset = (i - norm_inizio) * lunghezza_raw // lunghezza_norm
                    posizioni[i] = raw_inizio + min(offset, lunghezza_raw - 1)
                elif testo:
                    posizioni[i] = min(raw_inizio, len(testo) - 1)
    mappate = [p for p in posizioni[inizio:fine] if p is not None]
    return testo[min(mappate):max(mappate) + 1] if mappate else ""


def verifica_citazione(citazione, testo, soglia):
    """(esito, punteggio, tratto del testo originale, trovata) per una citazione.

    esatta: compare nel testo dopo la normalizzazione; altrimenti rapidfuzz.partial_ratio:
    approssimata se il punteggio è almeno la soglia, non_trovata sotto.
    """
    citazione = normalizza(citazione).strip('"\' ')
    testo_normalizzato = normalizza(testo)
    if not citazione:
        return "non_trovata", 0.0, "", False
    inizio = testo_normalizzato.find(citazione)
    if inizio >= 0:
        tratto = tratto_originale(testo, testo_normalizzato, inizio, inizio + len(citazione))
        return "esatta", 100.0, tratto, True
    allineamento = fuzz.partial_ratio_alignment(citazione, testo_normalizzato)
    punteggio = float(allineamento.score)
    tratto = tratto_originale(testo, testo_normalizzato, allineamento.dest_start, allineamento.dest_end)
    if punteggio >= soglia:
        return "approssimata", punteggio, tratto, True
    return "non_trovata", punteggio, tratto, False


def citazioni_non_vuote(citazioni):
    """Le citazioni non vuote: stringa singola (log attuali) o lista (primi log)."""
    if citazioni is None:
        return []
    if isinstance(citazioni, str):
        citazioni = [citazioni]
    return [str(c).strip() for c in citazioni if str(c).strip()]


def nome_file(s):
    """Rende sicuro un nome di modello/strategia per usarlo in un nome di file."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)
