"""Passo 2: assembla i messaggi per una singola risposta.

Usato da 03_run.py. Lanciato da solo, stampa il prompt per la prima risposta di
valutazione.csv, così puoi controllare cosa riceve il modello.
"""
import pandas as pd

from common import DOMANDA, PROMPTS, SPLITS, carica_config


def codebook(cfg):
    """Le categorie di config.yaml come elenco puntato, da inserire in system.txt."""
    righe = [f"- {nome}: {definizione}" for nome, definizione in cfg["categorie"].items()]
    righe.append(f"- {cfg['non_classificabile']}: la risposta è vuota, incomprensibile, fuori tema "
                 "(es. 'boh', 'non so', '.'), oppure non rientra chiaramente in nessuna delle categorie "
                 "precedenti. Usala invece di forzare una categoria.")
    return "\n".join(righe)


def blocco_esempi(strategia):
    """Testo fisso degli esempi (prompts/esempi/<domanda>/esempi_<strategia>.csv, da 01b_esempi.py):
    solo risposta e categoria umana, raggruppati per categoria nell'ordine del codebook."""
    esempi = pd.read_csv(
        PROMPTS / "esempi" / DOMANDA / f"esempi_{strategia}.csv", dtype={"id": str}, encoding="utf-8-sig"
    ).sort_values("posizione")
    righe = [
        "# Esempi codificati",
        "",
        "Risposte alla stessa domanda, con la categoria assegnata dal codificatore umano. "
        "Gli esempi sono raggruppati per categoria; le categorie senza esempi restano valide.",
    ]
    for categoria, gruppo in esempi.groupby("categoria", sort=False):
        righe += ["", f"## {categoria}"]
        righe += [f'- "{" ".join(str(t).split())}"' for t in gruppo["testo"]]
    return "\n".join(righe)


def costruisci_messaggi(testo, strategia, cfg):
    # replace e non format: il prompt contiene le graffe dell'esempio JSON
    system = (PROMPTS / "system.txt").read_text(encoding="utf-8").replace("{codebook}", codebook(cfg))
    domanda = cfg["dati"]["domande"][DOMANDA]["testo_domanda"]
    utente = f'Domanda: "{domanda}"\n\nRisposta da codificare:\n"""\n{testo}\n"""'

    if strategia == "zero_shot":
        contenuto = utente
    elif strategia in ("one_shot", "few_shot"):
        # Esempi in un blocco a parte, identico in ogni chiamata: system + esempi formano un
        # prefisso fisso che si può mettere in cache senza cambiare il testo.
        contenuto = [
            {"type": "text", "text": blocco_esempi(strategia)},
            {"type": "text", "text": utente},
        ]
    else:
        raise NotImplementedError(f"Strategia '{strategia}' non implementata.")
    return [{"role": "system", "content": system}, {"role": "user", "content": contenuto}]


def testo_messaggio(contenuto):
    """Il contenuto di un messaggio come testo, anche quando è diviso in blocchi."""
    if isinstance(contenuto, str):
        return contenuto
    return "\n\n".join(blocco["text"] for blocco in contenuto)


if __name__ == "__main__":
    cfg = carica_config()
    strategia = cfg["strategie"][0]
    riga = pd.read_csv(SPLITS / "valutazione.csv", encoding="utf-8-sig").iloc[0]
    prompt = "\n".join(
        f"===== {m['role']} =====\n{testo_messaggio(m['content'])}\n"
        for m in costruisci_messaggi(riga["testo"], strategia, cfg)
    )
    print(prompt)
    percorso = PROMPTS / "esempi" / DOMANDA / f"prompt_esempio_{strategia}.txt"
    percorso.parent.mkdir(parents=True, exist_ok=True)
    percorso.write_text(prompt, encoding="utf-8")
    print(f"Scritto {percorso}")
