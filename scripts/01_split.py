"""Passo 1: dall'Excel originale a serbatoio / valutazione / resto, con seed fisso.

Lavora sulla domanda indicata da dati.domanda_attiva in config.yaml.
Output (con colonne standardizzate id, testo, umano):
  data/splits/<domanda>/serbatoio.csv, valutazione.csv, resto.csv
"""
import pandas as pd
from ftfy import fix_text
from openpyxl import load_workbook
from openpyxl.utils import column_index_from_string

from common import DOMANDA, RESULTS, ROOT, SPLITS, carica_config, tutte_le_etichette

cfg = carica_config()
d = cfg["dati"]
q = d["domande"][DOMANDA]

percorso_excel = ROOT / d["excel"]
df = pd.read_excel(percorso_excel, sheet_name=d["foglio"])
print(f"Domanda: {DOMANDA} ({q['testo_domanda']})")
print(f"Righe nell'Excel: {len(df)}")

sequenze_mojibake = ("Ã", "â€", "Â")
modifiche_mojibake = []
residui_mojibake = []
for nome_domanda, dettagli in d["domande"].items():
    indice_colonna = column_index_from_string(dettagli["col_testo"]) - 1
    for indice_riga, valore in enumerate(df.iloc[:, indice_colonna]):
        if pd.isna(valore):
            continue
        testo_prima = str(valore)
        ha_mojibake = any(sequenza in testo_prima for sequenza in sequenze_mojibake)
        testo_dopo = fix_text(testo_prima) if ha_mojibake else testo_prima
        if testo_dopo != testo_prima:
            df.iat[indice_riga, indice_colonna] = testo_dopo
            modifiche_mojibake.append({
                "id": indice_riga + 2,
                "domanda": nome_domanda,
                "testo_prima": testo_prima,
                "testo_dopo": testo_dopo,
            })
        if any(sequenza in testo_dopo for sequenza in sequenze_mojibake):
            residui_mojibake.append((indice_riga + 2, nome_domanda))

if residui_mojibake:
    esempi = ", ".join(f"ID {id_} ({domanda})" for id_, domanda in residui_mojibake[:10])
    raise ValueError(f"Mojibake ancora presente dopo ftfy: {esempi}")

if modifiche_mojibake:
    workbook = load_workbook(percorso_excel)
    foglio = workbook[d["foglio"]]
    for modifica in modifiche_mojibake:
        indice_colonna = column_index_from_string(
            d["domande"][modifica["domanda"]]["col_testo"]
        )
        foglio.cell(row=modifica["id"], column=indice_colonna).value = modifica["testo_dopo"]
    percorso_corretto = percorso_excel.with_name(
        f"{percorso_excel.stem}_ftfy{percorso_excel.suffix}"
    )
    workbook.save(percorso_corretto)
    RESULTS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(modifiche_mojibake).to_csv(
        RESULTS / "correzioni_mojibake.csv", index=False, encoding="utf-8-sig"
    )
    print(f"Dataset originale intatto; copia corretta: {percorso_corretto}")
    print(f"Log correzioni: {RESULTS / 'correzioni_mojibake.csv'}")
else:
    print("Controllo mojibake: nessuna sequenza sospetta dopo ftfy.")


def colonna(lettera):
    return df.iloc[:, column_index_from_string(lettera) - 1]


df = pd.DataFrame({
    "id": df.index + 2,  # numero di riga in Excel (riga 1 = intestazioni): così si ritrova la risposta nel file
    "testo": colonna(q["col_testo"]),
    "umano": colonna(q["col_umano"]),
})

# Scarta risposte vuote (non hanno senso né per il modello né per il confronto)
vuote = df["testo"].isna() | (df["testo"].astype(str).str.strip() == "")
if vuote.any():
    print(f"Scarto {vuote.sum()} risposte vuote.")
df = df[~vuote]

# Prima codifica "-" = non ancora codificata: senza riferimento umano non si può confrontare
non_codificate = df["umano"].astype(str).str.strip() == d["non_codificato"]
if non_codificate.any():
    print(f"Scarto {non_codificate.sum()} risposte non ancora codificate ('{d['non_codificato']}').")
df = df[~non_codificate]

# Cella vuota nella prima codifica = non classificabile (vuoti, non so, rumore)
df["umano"] = df["umano"].fillna(cfg["non_classificabile"]).astype(str).str.strip()

# Etichette umane che non compaiono nel codebook: meglio saperlo subito
ignote = set(df["umano"]) - set(tutte_le_etichette(cfg))
if ignote:
    print(f"ATTENZIONE: etichette umane non presenti in config.yaml: {sorted(ignote)}")

print(f"Risposte utilizzabili: {len(df)}")
print("Distribuzione della codifica umana:")
print(df["umano"].value_counts().to_string())

# Mescola con seed fisso e taglia
df = df.sample(frac=1, random_state=cfg["seed"]).reset_index(drop=True)
n_s, n_v = cfg["split"]["n_serbatoio"], cfg["split"]["n_valutazione"]
assert n_s + n_v <= len(df), f"Servono almeno {n_s + n_v} risposte, ne ho {len(df)}."

serbatoio = df.iloc[:n_s]
valutazione = df.iloc[n_s:n_s + n_v]
resto = df.iloc[n_s + n_v:]

# Verifica che non si sovrappongano
ids = [set(serbatoio["id"]), set(valutazione["id"]), set(resto["id"])]
assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2]), "Split sovrapposti!"

SPLITS.mkdir(parents=True, exist_ok=True)
for nome, parte in [("serbatoio", serbatoio), ("valutazione", valutazione), ("resto", resto)]:
    parte.to_csv(SPLITS / f"{nome}.csv", index=False, encoding="utf-8-sig")
    print(f"{nome:12s} {len(parte):5d} risposte")
