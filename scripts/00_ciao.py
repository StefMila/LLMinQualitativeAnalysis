"""Fase 0: una chiamata di prova per ogni modello in config.yaml.

Verifica chiavi, endpoint e SDK, e che il modello restituisca il JSON atteso
codificando una risposta inventata con il prompt vero.
"""
import importlib

from dotenv import load_dotenv

from common import ROOT, carica_config, tutte_le_etichette

run = importlib.import_module("03_run")
costruisci_messaggi = importlib.import_module("02_prompt").costruisci_messaggi

load_dotenv(ROOT / ".env", encoding="utf-8")
cfg = carica_config()
etichette = tutte_le_etichette(cfg)
clienti = run.crea_clienti(cfg)

testo = "costruire il robot con i miei compagni"
messaggi = costruisci_messaggi(testo, cfg["strategie"][0], cfg)

for m in cfg["modelli"]:
    print(f"===== {m['provider']} / {m['nome']} =====")
    try:
        grezzo, versione, token = run.chiama(m, clienti, messaggi, cfg, testo, etichette)
        print("Versione effettiva:", versione)
        print("Token:             ", token)
        print("Risposta:          ", grezzo)
        print("JSON valido:       ", run.valida(grezzo, etichette))
    except Exception as e:
        print(f"ERRORE: {type(e).__name__}: {e}")
