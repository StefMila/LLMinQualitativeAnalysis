"""Stampa le risposte a cui le run valide dell'esperimento attivo assegnano categorie diverse,
con testo, codifica umana, motivazione e citazione di ogni run. Prima la valutazione, poi il serbatoio.

Uso: python scripts/discordanti.py    Output: results/<domanda>/check_<esperimento_id>.txt
"""
from collections import defaultdict

from common import RESULTS, carica_config, leggi_calls, leggi_split, nome_file

esperimento_id = carica_config()["esperimento_id"]
SET = ("valutazione", "serbatoio")  # ordine del report
split = {nome: leggi_split(nome).set_index("id") for nome in SET}

# Per ogni run il record più recente (una run ritentata ha più record)
casi = defaultdict(dict)
for r in leggi_calls():
    if r.get("esperimento_id") != esperimento_id:
        continue
    chiave = (SET.index(r.get("set", "valutazione")), r["provider"], r["modello"], r["strategia"], str(r["id"]))
    precedente = casi[chiave].get(r["run"])
    if precedente is None or r["ts"] >= precedente["ts"]:
        casi[chiave][r["run"]] = r

righe = []
for (indice_set, provider, modello, strategia, id_), runs in sorted(casi.items()):
    valide = sorted((r for r in runs.values() if r["valido"]), key=lambda r: r["run"])
    if len({r["categoria"] for r in valide}) < 2:
        continue
    risposta = split[SET[indice_set]].loc[id_]
    righe += ["", "=" * 80, f"{SET[indice_set]} | {provider}/{modello} | {strategia} | risposta {id_}",
              f"TESTO:  {risposta['testo']}", f"UMANO:  {risposta['umano']}"]
    for r in valide:
        righe += ["", f"  RUN {r['run']}: {r['categoria']}", f"  MOTIVAZIONE: {r['motivazione']}",
                  f"  QUOTE:       {r['quote']}"]
if not righe:
    righe = [f"Nessuna risposta con categorie discordanti per {esperimento_id}."]

percorso = RESULTS / f"check_{nome_file(esperimento_id)}.txt"
RESULTS.mkdir(parents=True, exist_ok=True)
contenuto = "\n".join(righe) + f"\n\nReport salvato in: {percorso}\n"
percorso.write_text(contenuto, encoding="utf-8")
print(contenuto, end="")
