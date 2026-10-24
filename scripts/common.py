"""Funzioni condivise da tutti gli script."""
import json
import re
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

from ftfy import fix_text
from rapidfuzz import fuzz
import yaml

# Su Windows la console non è UTF-8: senza questo accenti e "κ" fanno errore
sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
PROMPTS = ROOT / "prompts"

# Chiave che identifica una singola chiamata: serve per riprendere dopo un'interruzione.
CHIAVE = ["esperimento_id", "set", "provider", "modello", "strategia", "run", "id"]

# Gruppi di stabilità (unica definizione), sulla quota modale = run con l'etichetta modale / run previste:
#   stabile            10/10
#   abbastanza_stabile 8-9/10
#   instabile          7/10 o meno, comprese le risposte senza etichetta modale
# Valore = quota minima del gruppo; una risposta va nel primo gruppo con quota >= minimo.
STABILE = "stabile"
ABBASTANZA_STABILE = "abbastanza_stabile"
INSTABILE = "instabile"
GRUPPI_STABILITA = {STABILE: 1.0, ABBASTANZA_STABILE: 0.8, INSTABILE: 0.0}


def carica_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Split e risultati sono separati per domanda: cambiando domanda_attiva non si mescolano.
DOMANDA = carica_config()["dati"]["domanda_attiva"]
SPLITS = ROOT / "data" / "splits" / DOMANDA
RESULTS = ROOT / "results" / DOMANDA
CALLS = RESULTS / "calls.jsonl"
METRICS = RESULTS / "metrics"


def tutte_le_etichette(cfg):
    return list(cfg["categorie"]) + [cfg["non_classificabile"]]


def leggi_calls():
    """Restituisce la lista di record in calls.jsonl (vuota se il file non esiste)."""
    if not CALLS.exists():
        return []
    with open(CALLS, encoding="utf-8") as f:
        return [json.loads(riga) for riga in f if riga.strip()]


def normalizza(testo):
    """Minuscole, spazi compattati, virgolette tipografiche uniformate: per confrontare le quote."""
    testo = unicodedata.normalize("NFKC", fix_text(str(testo))).lower()
    testo = testo.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    testo = testo.replace("“", '"').replace("”", '"').replace("„", '"').replace("‟", '"')
    return re.sub(r"\s+", " ", testo).strip()


def tratto_originale(testo, testo_normalizzato, inizio, fine):
    """Mappa uno span normalizzato al corrispondente tratto del testo raw."""
    if inizio >= fine or not testo_normalizzato:
        return ""
    posizioni = [None] * len(testo_normalizzato)
    matcher = SequenceMatcher(None, testo, testo_normalizzato, autojunk=False)
    for operazione, raw_inizio, raw_fine, norm_inizio, norm_fine in matcher.get_opcodes():
        if operazione == "equal":
            for indice in range(norm_inizio, norm_fine):
                posizioni[indice] = raw_inizio + indice - norm_inizio
        elif operazione in ("replace", "insert") and norm_fine > norm_inizio:
            lunghezza_raw = raw_fine - raw_inizio
            lunghezza_norm = norm_fine - norm_inizio
            for indice in range(norm_inizio, norm_fine):
                if lunghezza_raw:
                    offset = (indice - norm_inizio) * lunghezza_raw // lunghezza_norm
                    posizioni[indice] = raw_inizio + min(offset, lunghezza_raw - 1)
                elif testo:
                    posizioni[indice] = min(raw_inizio, len(testo) - 1)
    mappate = [posizione for posizione in posizioni[inizio:fine] if posizione is not None]
    return testo[min(mappate):max(mappate) + 1] if mappate else ""


def verifica_citazione(citazione, testo, soglia):
    """Restituisce esito, punteggio, span raw e flag trovato per una citazione."""
    citazione_normalizzata = normalizza(citazione).strip('"\' ')
    testo_normalizzato = normalizza(testo)
    if not citazione_normalizzata:
        return "non_trovata", 0.0, "", False

    inizio_esatto = testo_normalizzato.find(citazione_normalizzata)
    if inizio_esatto >= 0:
        fine_esatto = inizio_esatto + len(citazione_normalizzata)
        return (
            "esatta",
            100.0,
            tratto_originale(testo, testo_normalizzato, inizio_esatto, fine_esatto),
            True,
        )

    allineamento = fuzz.partial_ratio_alignment(citazione_normalizzata, testo_normalizzato)
    punteggio = float(allineamento.score)
    tratto = tratto_originale(
        testo,
        testo_normalizzato,
        allineamento.dest_start,
        allineamento.dest_end,
    )
    if punteggio >= soglia:
        return "approssimata", punteggio, tratto, True
    return "non_trovata", punteggio, tratto, False


def citazioni_non_vuote(citazioni):
    """Restituisce citazioni non vuote, accettando liste storiche o stringhe singole."""
    if citazioni is None:
        return []
    if isinstance(citazioni, str):
        citazioni = [citazioni]
    return [str(citazione).strip() for citazione in citazioni if str(citazione).strip()]


def nome_file(s):
    """Rende sicuro un nome di modello/strategia per usarlo in un nome di file."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", s)
