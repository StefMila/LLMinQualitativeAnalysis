"""Stampa N output a caso da calls.jsonl accanto al testo e all'etichetta umana,
per leggere a mano come ragiona il modello.  Uso: python scripts/leggi.py [N]
"""
import random
import sys

import pandas as pd

from common import SPLITS, carica_config, leggi_calls

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
cfg = carica_config()
val = pd.read_csv(
    SPLITS / "valutazione.csv", dtype={"id": str}, encoding="utf-8-sig"
).set_index("id")

calls = leggi_calls()
random.seed(cfg["seed"])
for r in random.sample(calls, min(n, len(calls))):
    umano = val.loc[r["id"], "umano"]
    segno = "OK " if r["categoria"] == umano else "XX "
    print("=" * 80)
    print(f"{segno}id={r['id']}  run={r['run']}  {r['modello']} / {r['strategia']}")
    print(f"TESTO:       {val.loc[r['id'], 'testo']}")
    print(f"UMANO:       {umano}")
    print(f"LLM:         {r['categoria']}")
    print(f"MOTIVAZIONE: {r['motivazione']}")
    print(f"QUOTE:       {r['quote']}")
    if r["errori"]:
        print(f"ERRORI:      {r['errori']}")
