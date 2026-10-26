"""Passo 5: metriche di accordo con la codifica umana (Coding 1), solo sul campione di valutazione.

Output in results/<domanda>/metrics/:
  riepilogo.csv        una riga per esperimento, modello e strategia
  tabella_tesi.csv     colonne principali del riepilogo, solo esperimenti con 10 run
  per_categoria.csv    precision, recall e F1 per categoria
  stabilita.csv        accordo per gruppo di stabilità
  versioni.csv         legenda delle tre versioni delle metriche
  confusione_<esperimento>_<modello>_<strategia>.csv
κ di Cohen (con po e pe), accordo percentuale e F1 macro sono calcolati in tre versioni
(metriche.VERSIONI): nel riepilogo hanno il suffisso _principale, _controllo e _confronto_modelli,
negli altri file la colonna "versione".
"""
import pandas as pd
from sklearn.metrics import confusion_matrix

from common import CHIAVE, GRUPPI_STABILITA, METRICS, RESULTS, carica_config, nome_file, tabella_calls, tutte_le_etichette
from metriche import (
    SENZA_ETICHETTA, VERSIONI, accordo, fleiss_e_ta, gruppo_stabilita, id_con_etichetta_per_tutti,
    tasso_errori_formato,
)

RUN_TABELLA_TESI = 10
COLONNE_TABELLA_TESI = [
    "modello", "strategia", "copertura_principale", "kappa_cohen_principale",
    "accordo_percentuale_principale", "f1_macro_principale", "kappa_cohen_controllo",
    "kappa_cohen_confronto_modelli", "ta_at_n", "kappa_fleiss",
    "tasso_citazioni_approssimate", "tasso_citazioni_non_trovate",
]

cfg = carica_config()
etichette = tutte_le_etichette(cfg)
agg = pd.read_csv(RESULTS / "aggregated.csv", encoding="utf-8-sig")
agg["gruppo_stabilita"] = [gruppo_stabilita(e, q) for e, q in zip(agg["etichetta_modale"], agg["quota_modale"])]
citazioni = pd.read_csv(RESULTS / "citazioni.csv", encoding="utf-8-sig")
citazioni = citazioni[citazioni["set"] == "valutazione"]
calls = tabella_calls()
calls = calls[calls["set"] == "valutazione"]
calls_tutte = calls  # anche le chiamate poi rifatte: servono al tasso di errori di formato
calls = calls.drop_duplicates(CHIAVE, keep="last")
id_comuni = id_con_etichetta_per_tutti(agg)
METRICS.mkdir(parents=True, exist_ok=True)


def arrotonda(x):
    return round(float(x), 3) if pd.notna(x) else None


def gruppo(df, esperimento_id, modello, strategia):
    return df[(df["esperimento_id"] == esperimento_id) & (df["modello"] == modello) & (df["strategia"] == strategia)]


def date_esecuzione(chiamate):
    """"AAAA-MM-GG", oppure "prima - ultima" se l'esperimento è durato più giorni."""
    date = sorted(pd.to_datetime(chiamate["ts"], utc=True, errors="coerce").dropna().dt.strftime("%Y-%m-%d").unique())
    if not date:
        return None
    return date[0] if len(date) == 1 else f"{date[0]} - {date[-1]}"


riepilogo, per_categoria, stabilita = [], [], []
for (esperimento_id, modello, strategia), g in agg.groupby(["esperimento_id", "modello", "strategia"]):
    comuni = id_comuni[(esperimento_id, strategia)]
    acc = {versione: accordo(g, etichette, versione, comuni) for versione in VERSIONI}
    for versione, a in acc.items():
        for categoria, precisione, richiamo, f1, supporto in zip(
            etichette, a["precision"], a["recall"], a["f1"], a["supporto"]
        ):
            per_categoria.append({
                "esperimento_id": esperimento_id, "modello": modello, "strategia": strategia,
                "versione": versione, "categoria": categoria,
                "precision": round(precisione, 3), "recall": round(richiamo, 3), "f1": round(f1, 3),
                "supporto": int(supporto), "in_f1_macro": bool(supporto > 0),
            })

    righe_stabilita = []
    for nome_gruppo in GRUPPI_STABILITA:
        sotto = g[g["gruppo_stabilita"] == nome_gruppo]
        for versione in VERSIONI:
            a = accordo(sotto, etichette, versione, comuni)
            righe_stabilita.append({
                "esperimento_id": esperimento_id, "modello": modello, "strategia": strategia,
                "gruppo_stabilita": nome_gruppo, "versione": versione,
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
    stabilita += righe_stabilita

    cit = gruppo(citazioni, esperimento_id, modello, strategia)
    n_citazioni = len(cit)
    n_approssimate = int((cit["esito"] == "approssimata").sum())
    n_non_trovate = int((~cit["trovata"].astype(bool)).sum())
    tasso_approssimate = n_approssimate / n_citazioni if n_citazioni else float("nan")
    tasso_non_trovate = n_non_trovate / n_citazioni if n_citazioni else float("nan")

    chiamate = gruppo(calls, esperimento_id, modello, strategia)
    errori_formato = tasso_errori_formato(gruppo(calls_tutte, esperimento_id, modello, strategia))
    # Run previste: n_run salvato nei record; i primi log non l'hanno e si usa la run più alta
    if "n_run" in chiamate and chiamate["n_run"].notna().any():
        run_attese = int(chiamate["n_run"].dropna().max())
    else:
        run_attese = int(chiamate["run"].max())
    # κ di Fleiss e TA@R solo sulle risposte con tutte le run valide
    valide = chiamate[chiamate["valido"] & chiamate["categoria"].notna()]
    n_run_per_risposta = valide.groupby("id")["run"].nunique()
    id_complete = n_run_per_risposta[n_run_per_risposta == run_attese].index
    complete = valide[valide["id"].isin(id_complete)]
    conteggi_run = pd.crosstab(complete["id"], complete["categoria"]).reindex(columns=etichette, fill_value=0)
    kappa_fleiss, ta_at_n = fleiss_e_ta(conteggi_run, run_attese)

    # Matrice della versione "controllo": l'ultima colonna conta le risposte senza etichetta modale,
    # togliendola si ottiene la matrice della versione "principale".
    etichette_cm = etichette + [SENZA_ETICHETTA]
    a = acc["controllo"]
    cm = pd.DataFrame(
        confusion_matrix(a["y_umano"], a["y_llm"], labels=etichette_cm)[:-1] if a["n"] else 0,
        index=[f"umano: {e}" for e in etichette],
        columns=[f"llm: {e}" for e in etichette_cm],
    )
    cm.to_csv(METRICS / f"confusione_{nome_file(esperimento_id)}_{nome_file(modello)}_{nome_file(strategia)}.csv",
              encoding="utf-8-sig")

    n_senza_etichetta = int(g["etichetta_modale"].isna().sum())
    riga = {
        "esperimento_id": esperimento_id,
        "data_esecuzione": date_esecuzione(chiamate),
        "modello": modello,
        "strategia": strategia,
        "n_risposte": len(g),
        "n_senza_etichetta": n_senza_etichetta,
    }
    for versione, a in acc.items():
        if versione != "controllo":  # il controllo usa tutte le risposte: n = n_risposte, copertura 1
            riga |= {f"n_{versione}": a["n"], f"copertura_{versione}": arrotonda(a["copertura"])}
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
        "ta_at_n": arrotonda(ta_at_n),
        "kappa_fleiss": arrotonda(kappa_fleiss),
        "n_citazioni": n_citazioni,
        "n_citazioni_approssimate": n_approssimate,
        "tasso_citazioni_approssimate": arrotonda(tasso_approssimate),
        "n_citazioni_non_trovate": n_non_trovate,
        "tasso_citazioni_non_trovate": arrotonda(tasso_non_trovate),
        "n_errori_formato": errori_formato["n_errori_formato"],
        "n_risposte_modello": errori_formato["n_risposte_modello"],
        "tasso_errori_formato": arrotonda(errori_formato["tasso_errori_formato"]),
        "fonte_errori_formato": errori_formato["fonte_errori_formato"],
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
        r = {s["versione"]: s for s in righe_stabilita if s["gruppo_stabilita"] == nome_gruppo}
        print(f"  {nome_gruppo:<18} n ={r['controllo']['n_gruppo']:>3} "
              f"| {r['principale']['accordo_percentuale']} / {r['controllo']['accordo_percentuale']}")
    print(f"Risposte con {run_attese} run valide: {len(id_complete)}")
    if n_citazioni:
        print(f"Citazioni approssimate: {n_approssimate}/{n_citazioni} ({tasso_approssimate:.3f})")
        print(f"Citazioni non trovate: {n_non_trovate}/{n_citazioni} ({tasso_non_trovate:.3f})")
    else:
        print("Citazioni approssimate: nessuna citazione da valutare")
        print("Citazioni non trovate: nessuna citazione da valutare")
    if errori_formato["n_risposte_modello"]:
        print(f"Errori di formato: {errori_formato['n_errori_formato']}/{errori_formato['n_risposte_modello']} "
              f"risposte del modello ({errori_formato['tasso_errori_formato']:.4f}) "
              f"| fonte: {errori_formato['fonte_errori_formato']}")
    else:
        print(f"Errori di formato: dato non disponibile (fonte: {errori_formato['fonte_errori_formato']})")

riepilogo = pd.DataFrame(riepilogo)
riepilogo.to_csv(METRICS / "riepilogo.csv", index=False, encoding="utf-8-sig")
# Tabella per la tesi: solo esperimenti con 10 run; i baseline del prototipo restano in riepilogo.csv
tabella_tesi = riepilogo.loc[riepilogo["n_run"] == RUN_TABELLA_TESI, COLONNE_TABELLA_TESI]
tabella_tesi.to_csv(METRICS / "tabella_tesi.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(per_categoria).to_csv(METRICS / "per_categoria.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(stabilita).to_csv(METRICS / "stabilita.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([{"versione": v, "descrizione": d} for v, d in VERSIONI.items()]).to_csv(
    METRICS / "versioni.csv", index=False, encoding="utf-8-sig")
print("\nVersioni:")
for versione, descrizione in VERSIONI.items():
    print(f"  {versione}: {descrizione}")
print()
for nome in ("riepilogo.csv", "per_categoria.csv", "stabilita.csv", "versioni.csv"):
    print(f"Scritto {METRICS / nome}")
print(f"Scritto {METRICS / 'tabella_tesi.csv'} ({len(tabella_tesi)} righe)")
