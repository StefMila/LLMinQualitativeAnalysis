"""Prove della prompt caching di Claude prima di exp04, solo su risposte del serbatoio.

    python scripts/prova_cache.py verifica      # 5 chiamate few-shot con cache: la lettura scatta?
    python scripts/prova_cache.py equivalenza   # 50 risposte x 10 run, senza e con cache
    python scripts/prova_cache.py analisi       # tabella di confronto e costi

Le chiamate vanno in results/<domanda>/prova_cache/calls.jsonl (non in calls.jsonl, così non
entrano in aggregazioni e metriche). Il disegno è quello di 03_run.py: chiamate sequenziali e
indipendenti, stesso prompt, stessa temperatura e livello di ragionamento; cambia solo cache_control.
Rilanciando si saltano le chiamate già riuscite.
"""
import importlib
import json
import sys
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from common import RESULTS, ROOT, carica_config, leggi_calls, leggi_split, tutte_le_etichette
from metriche import etichetta_modale, fleiss_e_ta

run = importlib.import_module("03_run")
costruisci_messaggi = importlib.import_module("02_prompt").costruisci_messaggi

CARTELLA = RESULTS / "prova_cache"
LOG = CARTELLA / "calls.jsonl"
STRATEGIA = "few_shot"
N_VERIFICA = 5
N_RISPOSTE = 50
N_RUN = 10

# USD per milione di token (forniti il 2026-10-23)
PREZZI = {
    "anthropic": {"input": 5.0, "output": 25.0, "cache_scrittura": 6.25, "cache_lettura": 0.50},
    "openai": {"input": 0.20, "output": 1.20, "cache_scrittura": 0.20, "cache_lettura": 0.02},
}


def costo(record):
    """Costo in USD di un record: l'input non in cache è token_in meno scrittura e lettura."""
    p = PREZZI[record["provider"]]
    scrittura = record.get("token_cache_scrittura", 0)
    lettura = record.get("token_cache_lettura", 0)
    if record["provider"] == "openai":
        scrittura = 0  # nessun prezzo distinto per la scrittura: conta come input normale
    normale = record["token_in"] - scrittura - lettura
    return (
        normale * p["input"] + scrittura * p["cache_scrittura"] + lettura * p["cache_lettura"]
        + record["token_out"] * p["output"]
    ) / 1e6


def esegui(lavori, esperimento_id, cfg, etichette):
    """lavori = [(run, riga, cache)]; scrive un record per chiamata, come 03_run.py."""
    modello = next(m for m in cfg["modelli"] if m["provider"] == "anthropic")
    clienti = run.crea_clienti({"modelli": [modello]})
    fatte = {
        (r["run"], r["id"], r["cache_enabled"])
        for r in leggi_calls(LOG) if r["esperimento_id"] == esperimento_id and r["valido"]
    }
    lavori = [l for l in lavori if (l[0], str(l[1].id), l[2]) not in fatte]
    print(f"{esperimento_id}: da fare {len(lavori)} chiamate")
    CARTELLA.mkdir(parents=True, exist_ok=True)
    nuovi = []
    with open(LOG, "a", encoding="utf-8") as f:
        for i, (n_run, riga, cache) in enumerate(lavori, 1):
            messaggi = costruisci_messaggi(riga.testo, STRATEGIA, cfg)
            errori, out, grezzo, versione = [], None, None, None
            token = dict.fromkeys(run.CAMPI_TOKEN, 0)
            for tentativo in range(1, cfg["max_tentativi"] + 1):
                try:
                    grezzo, versione, t = run.chiama(
                        modello, clienti, messaggi, cfg, riga.testo, etichette, cache=cache
                    )
                    token = {k: token[k] + t[k] for k in run.CAMPI_TOKEN}
                    out = run.valida(grezzo, etichette)
                    break
                except ValueError as e:
                    errori.append(str(e))
                except Exception as e:
                    errori.append(f"{type(e).__name__}: {e}")
                    time.sleep(2 ** tentativo)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "provider": modello["provider"],
                "modello": modello["nome"],
                "modello_versione": versione,
                "esperimento_id": esperimento_id,
                "set": "serbatoio",
                "domanda": cfg["dati"]["domanda_attiva"],
                "temperatura": cfg["temperatura"],
                "livello_ragionamento": cfg["livello_ragionamento"]["anthropic"],
                "cache_enabled": cache,
                "strategia": STRATEGIA,
                "run": n_run,
                "id": str(riga.id),
                "valido": out is not None,
                "categoria": out["categoria"] if out else None,
                "tentativi": len(errori) + (1 if out else 0),
                "errori": errori,
                "output_grezzo": grezzo,
                **token,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            nuovi.append(record)
            print(
                f"[{i}/{len(lavori)}] run={n_run} id={riga.id} cache={cache} -> {record['categoria']} "
                f"in={token['token_in']} scrittura={token['token_cache_scrittura']} "
                f"lettura={token['token_cache_lettura']} out={token['token_out']} "
                f"ragionamento={token['token_ragionamento']}"
            )
    return nuovi


def conteggi_run_complete(gruppo, etichette, run_attese):
    """Categorie per risposta, solo risposte con tutte le run valide (come in 05_metrics.py)."""
    valide = gruppo[gruppo["valido"]]
    n_per_id = valide.groupby("id")["run"].nunique()
    complete = valide[valide["id"].isin(n_per_id[n_per_id == run_attese].index)]
    return pd.crosstab(complete["id"], complete["categoria"]).reindex(columns=etichette, fill_value=0)


def analisi(etichette):
    log = pd.DataFrame(leggi_calls(LOG))
    log = log[log["esperimento_id"] == "prova_cache_equivalenza"]
    if log.empty:
        raise SystemExit("Nessuna chiamata di equivalenza nel log.")
    log["costo_usd"] = log.apply(lambda r: costo(r.to_dict()), axis=1)
    righe, modali = [], {}
    for cache, g in log.groupby("cache_enabled"):
        nome = "con cache" if cache else "senza cache"
        modali[cache] = {
            id_: etichetta_modale(list(gr.loc[gr["valido"], "categoria"]), N_RUN)["etichetta_modale"]
            for id_, gr in g.groupby("id")
        }
        conteggi = conteggi_run_complete(g, etichette, N_RUN)
        kappa, ta = fleiss_e_ta(conteggi, N_RUN)
        righe.append({
            "condizione": nome,
            "chiamate": len(g),
            "valide": int(g["valido"].sum()),
            "errori_formato": int(g["errori"].apply(len).sum()),
            "risposte_con_etichetta_modale": sum(v is not None for v in modali[cache].values()),
            "TA@10": round(ta, 3),
            "kappa_fleiss": round(kappa, 3),
            "risposte_complete": len(conteggi),
            "token_in_medi": round(g["token_in"].mean(), 1),
            "cache_scrittura_medi": round(g["token_cache_scrittura"].mean(), 1),
            "cache_lettura_medi": round(g["token_cache_lettura"].mean(), 1),
            "token_out_medi": round(g["token_out"].mean(), 1),
            "ragionamento_medi": round(g["token_ragionamento"].mean(), 1),
            "costo_totale_usd": round(g["costo_usd"].sum(), 4),
            "costo_per_chiamata_usd": round(g["costo_usd"].mean(), 5),
        })
    tabella = pd.DataFrame(righe)
    ids = sorted(set(modali.get(False, {})) & set(modali.get(True, {})))
    uguali = [i for i in ids if modali[False][i] == modali[True][i]]
    diverse = pd.DataFrame(
        [{"id": i, "senza_cache": modali[False][i], "con_cache": modali[True][i]} for i in ids if i not in uguali]
    )
    tabella.to_csv(CARTELLA / "confronto.csv", index=False, encoding="utf-8-sig")
    diverse.to_csv(CARTELLA / "modali_diverse.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print(tabella.T.to_string(header=False))
    print(f"\nStessa etichetta modale nelle due condizioni: {len(uguali)}/{len(ids)} "
          f"({len(uguali) / len(ids):.1%}); senza etichetta in entrambe contano come uguali")
    if not diverse.empty:
        print(diverse.to_string(index=False))


def main():
    load_dotenv(ROOT / ".env", encoding="utf-8")
    cfg = carica_config()
    etichette = tutte_le_etichette(cfg)
    fase = sys.argv[1] if len(sys.argv) > 1 else ""
    serbatoio, valutazione = leggi_split("serbatoio"), leggi_split("valutazione")
    assert not set(serbatoio["id"]) & set(valutazione["id"]), "serbatoio e valutazione si sovrappongono"

    if fase == "verifica":
        righe = list(serbatoio.head(N_VERIFICA).itertuples())
        nuovi = esegui([(1, r, True) for r in righe], "prova_cache_verifica", cfg, etichette)
        letture = [r["token_cache_lettura"] for r in nuovi[1:]]
        if nuovi and min(letture, default=0) == 0:
            print("ATTENZIONE: cache_read_input_tokens = 0 dopo la prima chiamata: cache non attiva.")
        else:
            print("Cache attiva: lettura > 0 dalla seconda chiamata.")
    elif fase == "equivalenza":
        righe = list(serbatoio.head(N_RISPOSTE).itertuples())
        lavori = [(n, r, cache) for n in range(1, N_RUN + 1) for r in righe for cache in (False, True)]
        esegui(lavori, "prova_cache_equivalenza", cfg, etichette)
        analisi(etichette)
    elif fase == "analisi":
        analisi(etichette)
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
