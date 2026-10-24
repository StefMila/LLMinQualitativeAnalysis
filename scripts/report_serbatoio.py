"""Report descrittivo del pilota sul serbatoio: risposte senza etichetta modale e gruppi di stabilità.

Non entra nelle metriche della tesi. Usa le stesse regole di 04_aggregate.py (metriche.py).
Output: results/<domanda>/serbatoio_stabilita.csv
"""
import pandas as pd

from common import CHIAVE, GRUPPI_STABILITA, RESULTS, carica_config, leggi_calls
from metriche import etichetta_modale, gruppo_stabilita

cfg = carica_config()
esperimento_id = cfg["esperimento_id"]

calls = pd.DataFrame(leggi_calls())
calls["esperimento_id"] = calls.get("esperimento_id", pd.Series(dtype=object)).fillna("legacy")
calls["set"] = calls.get("set", pd.Series(dtype=object)).fillna("valutazione")
calls = calls.drop_duplicates(CHIAVE, keep="last")
calls = calls[(calls["set"] == "serbatoio") & (calls["esperimento_id"] == esperimento_id)]
if calls.empty:
    raise SystemExit(f"Nessuna chiamata del serbatoio per {esperimento_id}")

righe = []
for (modello, strategia, id_), g in calls.groupby(["modello", "strategia", "id"]):
    n_run_attese = int(g["n_run"].max())
    moda = etichetta_modale(g.loc[g["valido"], "categoria"].tolist(), n_run_attese)
    righe.append({
        "modello": modello,
        "strategia": strategia,
        "id": id_,
        "etichetta_modale": moda["etichetta_modale"],
        "quota_modale": moda["quota_modale"],
        "gruppo_stabilita": gruppo_stabilita(moda["etichetta_modale"], moda["quota_modale"]),
        "pareggio": moda["pareggio"],
        "run_mancanti": moda["run_mancanti"],
        "distribuzione": dict(moda["conteggi"]),
    })
dettaglio = pd.DataFrame(righe)

riepilogo = dettaglio.groupby(["modello", "strategia"]).agg(
    risposte=("id", "size"),
    senza_etichetta=("etichetta_modale", lambda s: int(s.isna().sum())),
    pareggi=("pareggio", "sum"),
    run_mancanti=("run_mancanti", "sum"),
)
conteggi_gruppi = pd.crosstab(
    [dettaglio["modello"], dettaglio["strategia"]], dettaglio["gruppo_stabilita"]
).reindex(columns=list(GRUPPI_STABILITA), fill_value=0)
riepilogo = riepilogo.join(conteggi_gruppi).reset_index()
riepilogo.insert(0, "esperimento_id", esperimento_id)
riepilogo.to_csv(RESULTS / "serbatoio_stabilita.csv", index=False, encoding="utf-8-sig")

print(f"Serbatoio, {esperimento_id} (descrittivo, non entra nelle metriche)")
print(riepilogo.to_string(index=False))
senza = dettaglio[dettaglio["etichetta_modale"].isna()]
if len(senza):
    print("\nRisposte senza etichetta modale:")
    print(senza[["modello", "id", "quota_modale", "distribuzione"]].to_string(index=False))
print(f"\nScritto {RESULTS / 'serbatoio_stabilita.csv'}")
