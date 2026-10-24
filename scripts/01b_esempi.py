"""Passo 1b: sceglie gli esempi per one-shot e few-shot dal serbatoio, con una regola fissa.

Regola (non modificarla dopo aver visto i risultati):
  - bacino: solo il serbatoio, escluse le risposte con testo normalizzato (minuscole, spazi
    unificati) identico a una risposta del campione di valutazione;
  - per ogni categoria umana presente nel bacino, un ordine casuale delle sue risposte con
    config.yaml → esempi.seed (ordine di partenza: id crescente);
  - one-shot = il primo esempio di ogni categoria, few-shot = i primi 3 (estrazione annidata);
    con meno risposte si usano tutte quelle disponibili.
Output in prompts/esempi/<domanda>/:
  ordine_esempi.csv     ordine completo per categoria (rango 1, 2, ...)
  esempi_<strategia>.csv esempi usati da ciascuna strategia, con la posizione nel prompt
"""
import re

import pandas as pd

from common import DOMANDA, PROMPTS, SPLITS, carica_config, tutte_le_etichette

ESEMPI = PROMPTS / "esempi" / DOMANDA
N_ESEMPI = {"one_shot": 1, "few_shot": 3}


def normalizza_testo(testo):
    return re.sub(r"\s+", " ", str(testo).lower()).strip()


def ordine_esempi(serbatoio, valutazione, etichette, seed):
    """Restituisce (ordine completo, risposte escluse perché uguali a una di valutazione)."""
    testi_valutazione = set(valutazione["testo"].map(normalizza_testo))
    duplicato = serbatoio["testo"].map(normalizza_testo).isin(testi_valutazione)
    bacino = serbatoio[~duplicato]
    parti = []
    for categoria in etichette:  # ordine del codebook
        risposte = bacino[bacino["umano"] == categoria]
        if risposte.empty:
            continue
        risposte = risposte.sort_values("id", key=lambda s: s.astype(int))
        risposte = risposte.sample(frac=1, random_state=seed).reset_index(drop=True)
        parti.append(risposte.assign(categoria=categoria, rango=range(1, len(risposte) + 1)))
    ordine = pd.concat(parti, ignore_index=True)[["categoria", "rango", "id", "testo"]]
    return ordine, serbatoio[duplicato]


def esempi_strategia(ordine, strategia):
    """Primi N_ESEMPI[strategia] di ogni categoria, nell'ordine del codebook, con la posizione nel prompt."""
    esempi = ordine[ordine["rango"] <= N_ESEMPI[strategia]].reset_index(drop=True)
    return esempi.assign(posizione=range(1, len(esempi) + 1))


if __name__ == "__main__":
    cfg = carica_config()
    etichette = tutte_le_etichette(cfg)
    seed = cfg["esempi"]["seed"]
    serbatoio = pd.read_csv(SPLITS / "serbatoio.csv", dtype={"id": str}, encoding="utf-8-sig")
    valutazione = pd.read_csv(SPLITS / "valutazione.csv", dtype={"id": str}, encoding="utf-8-sig")

    ordine, esclusi = ordine_esempi(serbatoio, valutazione, etichette, seed)
    assert not set(ordine["id"]) & set(valutazione["id"]), "Un esempio è nel campione di valutazione!"
    ESEMPI.mkdir(parents=True, exist_ok=True)
    ordine.to_csv(ESEMPI / "ordine_esempi.csv", index=False, encoding="utf-8-sig")

    print(f"Serbatoio: {len(serbatoio)} risposte | seed esempi: {seed}")
    print(f"Escluse perché uguali a una risposta di valutazione: {len(esclusi)}")
    for riga in esclusi.itertuples():
        print(f"  id {riga.id} ({riga.umano}): {riga.testo!r}")
    for strategia in N_ESEMPI:
        esempi = esempi_strategia(ordine, strategia)
        esempi[["posizione", "id", "testo", "categoria"]].to_csv(
            ESEMPI / f"esempi_{strategia}.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n{strategia}: {len(esempi)} esempi -> {ESEMPI / f'esempi_{strategia}.csv'}")
    disponibili = ordine.groupby("categoria").size().reindex(etichette, fill_value=0)
    print(f"\nRisposte nel bacino ed esempi few-shot per categoria (massimo {N_ESEMPI['few_shot']}):")
    for categoria, n in disponibili.items():
        segnala = "" if n >= N_ESEMPI["few_shot"] else "  <- meno di 3" if n else "  <- nessun esempio"
        print(f"  {categoria:<20} {n:>3} nel bacino, {min(n, N_ESEMPI['few_shot'])} usati{segnala}")
