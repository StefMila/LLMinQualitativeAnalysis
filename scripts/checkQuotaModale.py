"""Mostra le risposte in cui le run valide assegnano categorie diverse.

Uso: python scripts/checkQuotaModale.py
Legge solo i record dell'esperimento_id attivo in config.yaml: prima la valutazione, poi il serbatoio.
"""
from collections import defaultdict
from contextlib import redirect_stdout
from io import StringIO

import pandas as pd

from common import RESULTS, SPLITS, carica_config, leggi_calls, nome_file

cfg = carica_config()
esperimento_id = cfg["esperimento_id"]
SET = ("valutazione", "serbatoio")  # ordine del report
split = {
    nome: pd.read_csv(SPLITS / f"{nome}.csv", dtype={"id": str}, encoding="utf-8-sig").set_index("id")
    for nome in SET
}

# Se una run ha più record (ad esempio dopo un errore), considera quello più recente.
casi = defaultdict(dict)
for record in leggi_calls():
    if record.get("esperimento_id") != esperimento_id:
        continue
    chiave = (
        SET.index(record.get("set", "valutazione")),
        record["provider"],
        record["modello"],
        record["strategia"],
        str(record["id"]),
    )
    run = record["run"]
    precedente = casi[chiave].get(run)
    if precedente is None or record["ts"] >= precedente["ts"]:
        casi[chiave][run] = record

report = StringIO()
with redirect_stdout(report):
    trovati = False
    for chiave, runs in sorted(casi.items()):
        valide = [record for record in runs.values() if record["valido"]]
        categorie = {record["categoria"] for record in valide}
        if len(categorie) < 2:
            continue

        trovati = True
        indice_set, provider, modello, strategia, risposta_id = chiave
        set_nome = SET[indice_set]
        riga = split[set_nome].loc[risposta_id]
        print("\n" + "=" * 80)
        print(f"{set_nome} | {provider}/{modello} | {strategia} | risposta {risposta_id}")
        print(f"TESTO:  {riga['testo']}")
        print(f"UMANO:  {riga['umano']}")
        for record in sorted(valide, key=lambda elemento: elemento["run"]):
            print(f"\n  RUN {record['run']}: {record['categoria']}")
            print(f"  MOTIVAZIONE: {record['motivazione']}")
            print(f"  QUOTE:       {record['quote']}")

    if not trovati:
        print(f"Nessuna risposta con categorie discordanti per {esperimento_id}.")

report_path = RESULTS / f"check_{nome_file(esperimento_id)}.txt"
RESULTS.mkdir(parents=True, exist_ok=True)
contenuto = report.getvalue() + f"\nReport salvato in: {report_path}\n"
report_path.write_text(contenuto, encoding="utf-8")
print(contenuto, end="")
