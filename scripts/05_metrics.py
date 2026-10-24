"""Passo 5: metriche di accordo e classificazione rispetto alla codifica umana.

Output in results/<domanda>/metrics/: riepilogo.csv, tabella_tesi.csv (solo esperimenti con 10 run),
per_categoria.csv, stabilita.csv, versioni.csv e matrici di confusione.
κ di Cohen (con po e pe), accordo percentuale e F1 macro sono calcolati in tre versioni
(metriche.VERSIONI): nel riepilogo hanno il suffisso _principale, _controllo e _confronto_modelli,
negli altri file la colonna "versione". I log del serbatoio non entrano: aggregated.csv contiene
solo la valutazione e qui si tengono solo le chiamate con set = "valutazione".
"""
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import (
    CHIAVE, GRUPPI_STABILITA, METRICS, RESULTS, carica_config, leggi_calls, nome_file, tutte_le_etichette
)
from metriche import SENZA_ETICHETTA, VERSIONI, accordo, gruppo_stabilita, id_con_etichetta_per_tutti

cfg = carica_config()
etichette = tutte_le_etichette(cfg)
agg = pd.read_csv(RESULTS / "aggregated.csv", sep=None, engine="python", encoding="utf-8-sig")
if "esperimento_id" not in agg:
    agg["esperimento_id"] = "legacy"
calls = pd.DataFrame(leggi_calls())
if not calls.empty:
    if "esperimento_id" not in calls:
        calls["esperimento_id"] = "legacy"
    else:
        calls["esperimento_id"] = calls["esperimento_id"].fillna("legacy")
    if "set" not in calls:
        calls["set"] = "valutazione"
    else:
        calls["set"] = calls["set"].fillna("valutazione")
    calls = calls.drop_duplicates(CHIAVE, keep="last")
    calls = calls[calls["set"] == "valutazione"]
citazioni = pd.read_csv(
    RESULTS / "citazioni.csv", sep=None, engine="python", encoding="utf-8-sig"
)
citazioni = citazioni[citazioni["set"] == "valutazione"]
METRICS.mkdir(parents=True, exist_ok=True)

RUN_TABELLA_TESI = 10
COLONNE_TABELLA_TESI = [
    "modello", "strategia", "copertura_principale", "kappa_cohen_principale",
    "accordo_percentuale_principale", "f1_macro_principale", "kappa_cohen_controllo",
    "kappa_cohen_confronto_modelli", "ta_at_n", "kappa_fleiss",
    "tasso_citazioni_approssimate", "tasso_citazioni_non_trovate",
]


def arrotonda(x):
    return round(float(x), 3) if pd.notna(x) else None


agg["gruppo_stabilita"] = [
    gruppo_stabilita(e, q) for e, q in zip(agg["etichetta_modale"], agg["quota_modale"])
]
id_comuni = id_con_etichetta_per_tutti(agg)

riepilogo = []
per_categoria = []
stabilita = []
for (esperimento_id, modello, strategia), g in agg.groupby(
    ["esperimento_id", "modello", "strategia"]
):
    comuni = id_comuni[(esperimento_id, strategia)]
    acc = {versione: accordo(g, etichette, versione, comuni) for versione in VERSIONI}
    for versione, a in acc.items():
        for categoria, precisione, richiamo, f1_categoria, n_categoria in zip(
            etichette, a["precision"], a["recall"], a["f1"], a["supporto"]
        ):
            per_categoria.append({
                "esperimento_id": esperimento_id,
                "modello": modello,
                "strategia": strategia,
                "versione": versione,
                "categoria": categoria,
                "precision": round(precisione, 3),
                "recall": round(richiamo, 3),
                "f1": round(f1_categoria, 3),
                "supporto": int(n_categoria),
                "in_f1_macro": bool(n_categoria > 0),
            })
    n_senza_etichetta = int(g["etichetta_modale"].isna().sum())

    for nome_gruppo in GRUPPI_STABILITA:
        sotto = g[g["gruppo_stabilita"] == nome_gruppo]
        for versione in VERSIONI:
            a = accordo(sotto, etichette, versione, comuni)
            stabilita.append({
                "esperimento_id": esperimento_id,
                "modello": modello,
                "strategia": strategia,
                "gruppo_stabilita": nome_gruppo,
                "versione": versione,
                "n_gruppo": len(sotto),
                "quota_risposte_gruppo": arrotonda(len(sotto) / len(g)) if len(g) else None,
                "n_senza_etichetta_gruppo": int(sotto["etichetta_modale"].isna().sum()),
                "n": a["n"],
                "accordo_percentuale": arrotonda(a["accordo_percentuale"]),
                "kappa_cohen": arrotonda(a["kappa"]),
                "po": arrotonda(a["po"]),
                "pe": arrotonda(a["pe"]),
                "f1_macro": arrotonda(a["f1_macro"]),
            })

    citazioni_gruppo = citazioni[
        (citazioni["esperimento_id"] == esperimento_id)
        & (citazioni["modello"] == modello)
        & (citazioni["strategia"] == strategia)
    ]
    n_citazioni = len(citazioni_gruppo)
    n_citazioni_approssimate = int((citazioni_gruppo["esito"] == "approssimata").sum())
    n_citazioni_non_trovate = int((~citazioni_gruppo["trovata"].astype(bool)).sum())
    tasso_citazioni_approssimate = (
        n_citazioni_approssimate / n_citazioni if n_citazioni else float("nan")
    )
    tasso_citazioni_non_trovate = (
        n_citazioni_non_trovate / n_citazioni if n_citazioni else float("nan")
    )

    run_gruppo = calls[
        (calls["esperimento_id"] == esperimento_id)
        & (calls["modello"] == modello)
        & (calls["strategia"] == strategia)
    ] if not calls.empty else pd.DataFrame()
    if not run_gruppo.empty and "ts" in run_gruppo:
        date_esecuzione = sorted(
            pd.to_datetime(run_gruppo["ts"], utc=True, errors="coerce")
            .dropna()
            .dt.strftime("%Y-%m-%d")
            .unique()
        )
        data_esecuzione = (
            date_esecuzione[0]
            if len(date_esecuzione) == 1
            else f"{date_esecuzione[0]} - {date_esecuzione[-1]}"
            if date_esecuzione
            else None
        )
    else:
        data_esecuzione = None
    if not run_gruppo.empty:
        run_attese = (
            int(run_gruppo["n_run"].dropna().max())
            if "n_run" in run_gruppo and run_gruppo["n_run"].notna().any()
            else int(run_gruppo["run"].max()) if "run" in run_gruppo else cfg["n_run"]
        )
        valide_run = run_gruppo[run_gruppo["valido"] & run_gruppo["categoria"].notna()]
        n_run_per_risposta = valide_run.groupby("id")["run"].nunique()
        id_complete = n_run_per_risposta[n_run_per_risposta == run_attese].index
        complete = valide_run[valide_run["id"].isin(id_complete)]
        conteggi_run = pd.crosstab(complete["id"], complete["categoria"]).reindex(
            columns=etichette, fill_value=0
        )
    else:
        run_attese = cfg["n_run"]
        id_complete = []
        conteggi_run = pd.DataFrame(columns=etichette)

    # κ di Fleiss solo sulle risposte con tutte le run valide
    if len(conteggi_run) and run_attese > 1:
        accordo_per_risposta = (
            (conteggi_run.pow(2).sum(axis=1) - run_attese)
            / (run_attese * (run_attese - 1))
        )
        accordo_osservato = accordo_per_risposta.mean()
        proporzioni_categoria = conteggi_run.sum(axis=0) / (len(conteggi_run) * run_attese)
        accordo_atteso = (proporzioni_categoria**2).sum()
        kappa_fleiss = (
            (accordo_osservato - accordo_atteso) / (1 - accordo_atteso)
            if accordo_atteso < 1 else float("nan")
        )
        ta_at_n = (conteggi_run.max(axis=1) == run_attese).mean()
    else:
        kappa_fleiss = float("nan")
        ta_at_n = float("nan")

    # Matrice della versione "controllo": l'ultima colonna conta le risposte senza etichetta modale,
    # togliendola si ottiene la matrice della versione "principale".
    etichette_cm = etichette + [SENZA_ETICHETTA]
    etichette_righe = [f"umano: {e}" for e in etichette]
    etichette_colonne = [f"llm: {e}" for e in etichette_cm]
    a = acc["controllo"]
    if a["n"]:
        cm = pd.DataFrame(
            confusion_matrix(a["y_umano"], a["y_llm"], labels=etichette_cm)[:-1],
            index=etichette_righe,
            columns=etichette_colonne,
        )
    else:
        cm = pd.DataFrame(0, index=etichette_righe, columns=etichette_colonne)
    cm.to_csv(
        METRICS / f"confusione_{nome_file(esperimento_id)}_{nome_file(modello)}_{nome_file(strategia)}.csv",
        encoding="utf-8-sig",
    )

    riga = {
        "esperimento_id": esperimento_id,
        "data_esecuzione": data_esecuzione,
        "modello": modello,
        "strategia": strategia,
        "n_risposte": len(g),
        "n_senza_etichetta": n_senza_etichetta,
    }
    for versione, a in acc.items():
        if versione != "controllo":  # il controllo usa tutte le risposte: n = n_risposte, copertura 1
            riga |= {
                f"n_{versione}": a["n"],
                f"copertura_{versione}": arrotonda(a["copertura"]),
            }
        riga |= {
            f"kappa_cohen_{versione}": arrotonda(a["kappa"]),
            f"po_{versione}": arrotonda(a["po"]),
            f"pe_{versione}": arrotonda(a["pe"]),
            f"accordo_percentuale_{versione}": arrotonda(a["accordo_percentuale"]),
            f"f1_macro_{versione}": arrotonda(a["f1_macro"]),
        }
    riga |= {
        # categorie con almeno un caso umano tra le risposte della versione principale
        "n_categorie_f1_macro": int((acc["principale"]["supporto"] > 0).sum()),
        "n_run": run_attese,
        "n_risposte_run_complete": len(id_complete),
        "ta_at_n": round(ta_at_n, 3) if pd.notna(ta_at_n) else None,
        "kappa_fleiss": round(kappa_fleiss, 3) if pd.notna(kappa_fleiss) else None,
        "n_citazioni": n_citazioni,
        "n_citazioni_approssimate": n_citazioni_approssimate,
        "tasso_citazioni_approssimate": (
            round(tasso_citazioni_approssimate, 3)
            if pd.notna(tasso_citazioni_approssimate) else None
        ),
        "n_citazioni_non_trovate": n_citazioni_non_trovate,
        "tasso_citazioni_non_trovate": (
            round(tasso_citazioni_non_trovate, 3)
            if pd.notna(tasso_citazioni_non_trovate) else None
        ),
    }
    riepilogo.append(riga)

    print(f"\n=== {esperimento_id} | {modello} / {strategia} ===")
    print(f"Risposte senza etichetta modale: {n_senza_etichetta}/{len(g)}")
    for versione, a in acc.items():
        print(
            f"[{versione}] κ di Cohen = {a['kappa']:.3f} (po = {a['po']:.3f}, pe = {a['pe']:.3f}) "
            f"| accordo percentuale = {a['accordo_percentuale']:.3f} "
            f"| F1 macro = {a['f1_macro']:.3f} | n = {a['n']}/{a['n_totale']} "
            f"(copertura {a['copertura']:.3f})"
        )
    print(f"TA@{run_attese} = {ta_at_n:.3f} | κ di Fleiss = {kappa_fleiss:.3f}")
    print("Accordo per gruppo di stabilità (accordo percentuale principale / controllo):")
    for nome_gruppo in GRUPPI_STABILITA:
        righe_gruppo = {
            r["versione"]: r for r in stabilita
            if (r["esperimento_id"], r["modello"], r["strategia"], r["gruppo_stabilita"])
            == (esperimento_id, modello, strategia, nome_gruppo)
        }
        print(
            f"  {nome_gruppo:<18} n ={righe_gruppo['controllo']['n_gruppo']:>3} "
            f"| {righe_gruppo['principale']['accordo_percentuale']} "
            f"/ {righe_gruppo['controllo']['accordo_percentuale']}"
        )
    print(f"Risposte con {run_attese} run valide: {len(id_complete)}")
    print(
        f"Citazioni approssimate: {n_citazioni_approssimate}/{n_citazioni} "
        f"({tasso_citazioni_approssimate:.3f})"
        if pd.notna(tasso_citazioni_approssimate)
        else "Citazioni approssimate: nessuna citazione da valutare"
    )
    print(
        f"Citazioni non trovate: {n_citazioni_non_trovate}/{n_citazioni} "
        f"({tasso_citazioni_non_trovate:.3f})"
        if pd.notna(tasso_citazioni_non_trovate)
        else "Citazioni non trovate: nessuna citazione da valutare"
    )
riepilogo = pd.DataFrame(riepilogo)
riepilogo.to_csv(METRICS / "riepilogo.csv", index=False, encoding="utf-8-sig")
# Tabella per la tesi: solo esperimenti con 10 run; i baseline del prototipo restano in riepilogo.csv
tabella_tesi = riepilogo.loc[riepilogo["n_run"] == RUN_TABELLA_TESI, COLONNE_TABELLA_TESI]
tabella_tesi.to_csv(METRICS / "tabella_tesi.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(per_categoria).to_csv(METRICS / "per_categoria.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(stabilita).to_csv(METRICS / "stabilita.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(
    [{"versione": v, "descrizione": d} for v, d in VERSIONI.items()]
).to_csv(METRICS / "versioni.csv", index=False, encoding="utf-8-sig")
print("\nVersioni:")
for versione, descrizione in VERSIONI.items():
    print(f"  {versione}: {descrizione}")
print(f"\nScritto {METRICS / 'riepilogo.csv'}")
print(f"Scritto {METRICS / 'per_categoria.csv'}")
print(f"Scritto {METRICS / 'stabilita.csv'}")
print(f"Scritto {METRICS / 'versioni.csv'}")
print(f"Scritto {METRICS / 'tabella_tesi.csv'} ({len(tabella_tesi)} righe)")
