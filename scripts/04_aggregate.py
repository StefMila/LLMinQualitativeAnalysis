"""Passo 4: etichetta modale per risposta e verifica delle citazioni.

Output in results/<domanda>/:
  aggregated.csv       una riga per esperimento, modello, strategia e risposta di valutazione:
                       etichetta_modale, quota_modale (run con l'etichetta modale / run previste),
                       gruppo_stabilita, distribuzione delle etichette nelle run e
                       quota_risposte_con_tutte_le_citazioni_verificate (quota di run valide in cui
                       tutte le citazioni si trovano nel testo)
  citazioni.csv        una riga per citazione e run valida, con esito e tratto più simile del testo
  citazioni_da_esaminare.csv, citazioni_da_esaminare_serbatoio.csv
                       citazioni non esatte, con una colonna giudizio_umano da compilare
  metrics/sensibilita_citazioni.csv
                       esiti delle citazioni al variare della soglia (descrittivo)
"""
import pandas as pd

from common import (
    CHIAVE, METRICS, RESULTS, carica_config, citazioni_non_vuote, leggi_split, tabella_calls, verifica_citazione
)
from metriche import etichetta_modale, gruppo_stabilita

cfg = carica_config()
verifica_cfg = cfg["verifica_citazioni"]
soglia = float(verifica_cfg["soglia_somiglianza"])

calls = tabella_calls()
if calls.empty:
    raise SystemExit("calls.jsonl è vuoto: lancia prima 03_run.py")
calls = calls.drop_duplicates(CHIAVE, keep="last")  # chiamata ritentata: conta l'ultimo tentativo

split = {nome: leggi_split(nome) for nome in ("valutazione", "serbatoio")}
testi = {nome: dict(zip(df["id"], df["testo"])) for nome, df in split.items()}
umani = dict(zip(split["valutazione"]["id"], split["valutazione"]["umano"]))

# ---------------------------------------------------------------- citazioni
righe_citazioni, quote_ok = [], []
for record in calls.to_dict("records"):
    trovate = []
    if record["valido"]:
        testo = testi[record["set"]].get(str(record["id"]), "")
        for indice, citazione in enumerate(citazioni_non_vuote(record["quote"]), 1):
            esito, punteggio, tratto, trovata = verifica_citazione(citazione, testo, soglia)
            trovate.append(trovata)
            righe_citazioni.append({
                "esperimento_id": record["esperimento_id"],
                "id": str(record["id"]),
                "set": record["set"],
                "provider": record["provider"],
                "modello": record["modello"],
                "strategia": record["strategia"],
                "run": record["run"],
                "indice_citazione": indice,
                "citazione": citazione,
                "punteggio_somiglianza": punteggio,
                "esito": esito,
                "versione_verifica": verifica_cfg["versione_verifica"],
                "tratto_simile": tratto,
                "trovata": trovata,
            })
    # None = non verificabile (chiamata non valida o senza citazioni, es. "non classificabile")
    quote_ok.append(all(trovate) if trovate else None)
calls["quote_ok"] = quote_ok

citazioni = pd.DataFrame(righe_citazioni, columns=[
    "esperimento_id", "id", "set", "provider", "modello", "strategia", "run", "indice_citazione",
    "citazione", "punteggio_somiglianza", "esito", "versione_verifica", "tratto_simile", "trovata",
])
citazioni.to_csv(RESULTS / "citazioni.csv", index=False, encoding="utf-8-sig")

for set_nome, nome in (("valutazione", "citazioni_da_esaminare.csv"),
                       ("serbatoio", "citazioni_da_esaminare_serbatoio.csv")):
    da_esaminare = citazioni[(citazioni["set"] == set_nome) & (citazioni["esito"] != "esatta")]
    da_esaminare = da_esaminare[["id", "set", "citazione", "tratto_simile", "punteggio_somiglianza", "esito"]]
    da_esaminare.assign(giudizio_umano="").to_csv(RESULTS / nome, index=False, encoding="utf-8-sig")

sensibilita = []
for (esperimento_id, set_nome, modello, strategia), g in citazioni.groupby(
    ["esperimento_id", "set", "modello", "strategia"]
):
    esatte = g["esito"] == "esatta"
    for soglia_test in verifica_cfg["soglie_sensibilita"]:
        approssimate = ~esatte & (g["punteggio_somiglianza"] >= soglia_test)
        sensibilita.append({
            "esperimento_id": esperimento_id,
            "set": set_nome,
            "modello": modello,
            "strategia": strategia,
            "soglia": soglia_test,
            "citazioni_totali": len(g),
            "esatte": int(esatte.sum()),
            "approssimate": int(approssimate.sum()),
            "non_trovate": int((~esatte & ~approssimate).sum()),
        })
METRICS.mkdir(parents=True, exist_ok=True)
pd.DataFrame(sensibilita).to_csv(METRICS / "sensibilita_citazioni.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- etichetta modale
# Run previste: n_run salvato nei record; i primi log non l'hanno e si usa la run più alta
run_massime = calls.groupby(["esperimento_id", "set", "modello", "strategia"])["run"].max()
righe = []
for (esperimento_id, modello, strategia, id_), g in calls[calls["set"] == "valutazione"].groupby(
    ["esperimento_id", "modello", "strategia", "id"]
):
    valide = g[g["valido"]]
    verificabili = valide["quote_ok"].dropna()
    if "n_run" in g and g["n_run"].notna().any():
        n_run_attese = int(g["n_run"].dropna().max())
    else:
        n_run_attese = int(run_massime.get((esperimento_id, "valutazione", modello, strategia), cfg["n_run"]))
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
        "testo": testi["valutazione"].get(str(id_)),
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
