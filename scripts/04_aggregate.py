"""Passo 4: aggregazioni di valutazione e verifica delle citazioni per set.

Colonne principali di results/<domanda>/aggregated.csv:
  etichetta_modale   categoria più frequente tra le run valide
  quota_modale       frazione di run (su tutte quelle fatte) che hanno dato l'etichetta modale
  gruppo_stabilita   stabile, abbastanza_stabile o instabile (common.GRUPPI_STABILITA)
  pareggio           True se due categorie hanno la stessa frequenza massima
    quota_risposte_con_tutte_le_citazioni_verificate
                                         frazione di run valide in cui TUTTE le citazioni si trovano nel testo
citazioni.csv       una riga per citazione e run valida, con esito e span raw simile
"""
import pandas as pd

from common import (
    CHIAVE,
    METRICS,
    RESULTS,
    SPLITS,
    carica_config,
    citazioni_non_vuote,
    leggi_calls,
    verifica_citazione,
)
from metriche import etichetta_modale, gruppo_stabilita

cfg = carica_config()
verifica_cfg = cfg["verifica_citazioni"]
versione_verifica = verifica_cfg["versione_verifica"]
soglia = float(verifica_cfg["soglia_somiglianza"])

calls = pd.DataFrame(leggi_calls())
if calls.empty:
    raise SystemExit("calls.jsonl è vuoto: lancia prima 03_run.py")

# Se una combinazione è stata ritentata, tieni l'ultimo tentativo
if "esperimento_id" not in calls:
    calls["esperimento_id"] = "legacy"
else:
    calls["esperimento_id"] = calls["esperimento_id"].fillna("legacy")
if "set" not in calls:
    calls["set"] = "valutazione"
else:
    calls["set"] = calls["set"].fillna("valutazione")
calls = calls.drop_duplicates(CHIAVE, keep="last")

testi_per_set = {}
for set_nome in ("valutazione", "serbatoio"):
    percorso = SPLITS / f"{set_nome}.csv"
    dati = pd.read_csv(percorso, dtype={"id": str}, encoding="utf-8-sig")
    testi_per_set[set_nome] = dict(zip(dati["id"], dati["testo"]))
val = pd.read_csv(SPLITS / "valutazione.csv", dtype={"id": str}, encoding="utf-8-sig")
umani = dict(zip(val["id"], val["umano"]))
run_massimi = calls.groupby(["esperimento_id", "set", "modello", "strategia"])["run"].max()


def esito_citazione(citazione, testo):
    esito, punteggio, tratto, trovata = verifica_citazione(citazione, testo, soglia)
    return esito, punteggio, tratto, trovata


def quote_trovate(riga):
    if not riga["valido"]:
        return None
    quote = citazioni_non_vuote(riga["quote"])
    if not quote:
        return None  # nessuna citazione (es. "non classificabile"): non verificabile
    set_nome = riga.get("set", "valutazione")
    testo = testi_per_set[set_nome][str(riga["id"])]
    return all(esito_citazione(q, testo)[3] for q in quote)


calls["quote_ok"] = calls.apply(quote_trovate, axis=1)

righe_citazioni = []
for record in calls.to_dict("records"):
    if not record["valido"]:
        continue
    quote = citazioni_non_vuote(record["quote"])
    set_nome = record.get("set", "valutazione")
    testo = testi_per_set[set_nome].get(str(record["id"]), "")
    for indice, citazione in enumerate(quote, 1):
        citazione = str(citazione).strip()
        if not citazione:
            continue
        esito, punteggio, tratto, trovata = esito_citazione(citazione, testo)
        righe_citazioni.append({
            "esperimento_id": record["esperimento_id"],
            "id": str(record["id"]),
            "set": set_nome,
            "provider": record["provider"],
            "modello": record["modello"],
            "strategia": record["strategia"],
            "run": record["run"],
            "indice_citazione": indice,
            "citazione": citazione,
            "punteggio_somiglianza": punteggio,
            "esito": esito,
            "versione_verifica": versione_verifica,
            "tratto_simile": tratto,
            "trovata": trovata,
        })

citazioni = pd.DataFrame(righe_citazioni, columns=[
    "esperimento_id", "id", "set", "provider", "modello", "strategia", "run",
    "indice_citazione", "citazione", "punteggio_somiglianza", "esito",
    "versione_verifica", "tratto_simile", "trovata",
])
citazioni.to_csv(RESULTS / "citazioni.csv", index=False, encoding="utf-8-sig")

colonne_revisione = [
    "id", "set", "citazione", "tratto_simile", "punteggio_somiglianza", "esito"
]
for set_nome, nome_file in (("valutazione", "citazioni_da_esaminare.csv"),
                            ("serbatoio", "citazioni_da_esaminare_serbatoio.csv")):
    da_esaminare = citazioni[
        (citazioni["set"] == set_nome) & (citazioni["esito"] != "esatta")
    ][colonne_revisione].copy()
    da_esaminare["giudizio_umano"] = ""
    da_esaminare.to_csv(RESULTS / nome_file, index=False, encoding="utf-8-sig")

sensibilita = []
citazioni_sensibilita = citazioni
for (esperimento_id, set_nome, modello, strategia), gruppo in citazioni_sensibilita.groupby(
    ["esperimento_id", "set", "modello", "strategia"]
):
    esatte = gruppo["esito"] == "esatta"
    for soglia_test in verifica_cfg["soglie_sensibilita"]:
        approssimate = (~esatte) & (gruppo["punteggio_somiglianza"] >= soglia_test)
        non_trovate = (~esatte) & (~approssimate)
        sensibilita.append({
            "esperimento_id": esperimento_id,
            "set": set_nome,
            "modello": modello,
            "strategia": strategia,
            "soglia": soglia_test,
            "citazioni_totali": len(gruppo),
            "esatte": int(esatte.sum()),
            "approssimate": int(approssimate.sum()),
            "non_trovate": int(non_trovate.sum()),
        })
METRICS.mkdir(parents=True, exist_ok=True)
pd.DataFrame(sensibilita).to_csv(
    METRICS / "sensibilita_citazioni.csv", index=False, encoding="utf-8-sig"
)

righe = []
calls_val = calls[calls["set"] == "valutazione"]
for (esperimento_id, modello, strategia, id_), g in calls_val.groupby(
    ["esperimento_id", "modello", "strategia", "id"]
):
    valide = g[g["valido"]]
    verificabili = valide["quote_ok"].dropna()
    n_run_attese = (
        int(g["n_run"].dropna().max())
        if "n_run" in g and g["n_run"].notna().any()
        else int(run_massimi.get((esperimento_id, g["set"].iloc[0], modello, strategia), cfg["n_run"]))
    )
    moda = etichetta_modale(valide["categoria"].tolist(), n_run_attese)
    righe.append({
        "esperimento_id": esperimento_id,
        "modello": modello,
        "strategia": strategia,
        "id": id_,
        "umano": umani.get(str(id_)),
        "etichetta_modale": moda["etichetta_modale"],
        "quota_modale": moda["quota_modale"],
        "gruppo_stabilita": gruppo_stabilita(moda["etichetta_modale"], moda["quota_modale"]),
        "pareggio": moda["pareggio"],
        "da_revisionare": moda["da_revisionare"],
        "n_run": len(g),
        "n_run_attese": n_run_attese,
        "run_mancanti": moda["run_mancanti"],
        "n_valide": len(valide),
        "distribuzione": dict(moda["conteggi"]),
        "quota_risposte_con_tutte_le_citazioni_verificate": (
            verificabili.astype(float).mean() if len(verificabili) else None
        ),
        "testo": testi_per_set["valutazione"].get(str(id_)),
    })

agg = pd.DataFrame(righe)
agg.to_csv(RESULTS / "aggregated.csv", index=False, encoding="utf-8-sig")
print(f"Scritto {RESULTS / 'aggregated.csv'} ({len(agg)} righe)")
print(f"Scritto {RESULTS / 'citazioni.csv'} ({len(citazioni)} citazioni)")
print(f"Scritto {RESULTS / 'citazioni_da_esaminare.csv'}")
print(f"Scritto {RESULTS / 'citazioni_da_esaminare_serbatoio.csv'}")
print(f"Scritto {METRICS / 'sensibilita_citazioni.csv'}")
print(agg.groupby(["esperimento_id", "modello", "strategia"])[
    ["quota_modale", "quota_risposte_con_tutte_le_citazioni_verificate"]
].mean().round(3))
