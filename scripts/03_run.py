"""Passo 3: chiama il modello per ogni combinazione modello x strategia x run x risposta.

Ogni chiamata viene aggiunta a results/<domanda>/calls.jsonl appena arriva. Rilanciando lo
script si saltano le combinazioni già riuscite (quelle fallite vengono ritentate).
"""
import copy
import importlib
import json
import random
import time
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from common import CALLS, ROOT, SPLITS, carica_config, leggi_calls, tutte_le_etichette

costruisci_messaggi = importlib.import_module("02_prompt").costruisci_messaggi


# ---------------------------------------------------------------- provider
# Ogni chiamata restituisce (testo, versione, token). In token:
#   token_in               tutti i token di input, in cache o no (come nei log di exp03)
#   token_out              tutti i token di output, ragionamento compreso
#   token_cache_scrittura  input scritti in cache (Claude: cache_creation_input_tokens;
#                          GPT su Azure: prompt_tokens_details.cache_write_tokens, se presente)
#   token_cache_lettura    input letti dalla cache (Claude: cache_read_input_tokens;
#                          GPT: prompt_tokens_details.cached_tokens)
#   token_ragionamento     parte di token_out spesa nel ragionamento (Claude:
#                          output_tokens_details.thinking_tokens; GPT: reasoning_tokens)
CAMPI_TOKEN = ["token_in", "token_out", "token_cache_scrittura", "token_cache_lettura", "token_ragionamento"]


def chiama_openai(client, modello, messaggi, temperatura, livello_ragionamento):
    extra = {} if temperatura is None else {"temperature": temperatura}
    risp = client.chat.completions.create(
        model=modello,
        messages=messaggi,
        response_format={"type": "json_object"},  # Fase 2: json_schema con enum delle categorie
        reasoning_effort=livello_ragionamento,
        **extra,
    )
    u = risp.usage
    dettagli_in = getattr(u, "prompt_tokens_details", None)
    dettagli_out = getattr(u, "completion_tokens_details", None)
    token = {
        "token_in": u.prompt_tokens,  # comprende già cached_tokens
        "token_out": u.completion_tokens,  # comprende già reasoning_tokens
        "token_cache_scrittura": getattr(dettagli_in, "cache_write_tokens", None) or 0,
        "token_cache_lettura": getattr(dettagli_in, "cached_tokens", None) or 0,
        "token_ragionamento": getattr(dettagli_out, "reasoning_tokens", None) or 0,
    }
    return risp.choices[0].message.content, risp.model, token


# Stesse chiavi che system.txt chiede a tutti i modelli; la categoria resta libera (niente enum)
# così Claude non è più vincolato di OpenAI, e valida() controlla che sia ammessa.
SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "motivazione": {"type": "string"},
        "quote": {"type": "string"},
        "categoria": {"type": "string"},
    },
    "required": ["motivazione", "quote", "categoria"],
    "additionalProperties": False,
}


def con_cache(system, messaggi):
    """Aggiunge cache_control alla fine del prefisso fisso, senza toccare il testo.

    Con gli esempi (one/few-shot) il prefisso è system + blocco degli esempi e il segnaposto va
    sull'ultimo blocco prima della risposta da codificare; in zero-shot va sul system prompt.
    """
    segnaposto = {"type": "ephemeral"}  # durata predefinita: 5 minuti
    messaggi = copy.deepcopy(messaggi)
    contenuto = messaggi[0]["content"]
    if isinstance(contenuto, list) and len(contenuto) > 1:
        contenuto[-2]["cache_control"] = segnaposto
        return system, messaggi
    return [{"type": "text", "text": system, "cache_control": segnaposto}], messaggi


def chiama_anthropic(client, modello, messaggi, temperatura, livello_ragionamento, cache=False):
    # Claude vuole il system prompt a parte, non dentro messages
    system = "\n\n".join(m["content"] for m in messaggi if m["role"] == "system")
    messaggi = [m for m in messaggi if m["role"] != "system"]
    if cache:
        system, messaggi = con_cache(system, messaggi)
    extra = {} if temperatura is None else {"temperature": temperatura}
    risp = client.messages.create(
        model=modello,
        max_tokens=16000,  # include il ragionamento interno (thinking), non solo il JSON
        system=system,
        messages=messaggi,
        output_config={
            "effort": livello_ragionamento,
            "format": {"type": "json_schema", "schema": SCHEMA_JSON},
        },
        **extra,
    )
    if risp.stop_reason == "refusal":
        raise ValueError(f"il modello ha rifiutato: {risp.stop_details}")
    testo = next((b.text for b in risp.content if b.type == "text"), "")
    u = risp.usage
    scrittura = getattr(u, "cache_creation_input_tokens", None) or 0
    lettura = getattr(u, "cache_read_input_tokens", None) or 0
    dettagli_out = getattr(u, "output_tokens_details", None)
    token = {
        # input_tokens di Claude esclude la cache: si somma per avere tutto l'input come in exp03
        "token_in": u.input_tokens + scrittura + lettura,
        "token_out": u.output_tokens,  # comprende già thinking_tokens
        "token_cache_scrittura": scrittura,
        "token_cache_lettura": lettura,
        "token_ragionamento": getattr(dettagli_out, "thinking_tokens", None) or 0,
    }
    return testo, risp.model, token


def chiama_mock(testo, etichette):
    """Finto modello: categoria a caso, quota = prime parole. Serve solo a provare la pipeline."""
    quota = " ".join(str(testo).split()[:4])
    out = {"motivazione": "risposta finta", "quote": quota, "categoria": random.choice(etichette)}
    return json.dumps(out, ensure_ascii=False), "mock", dict.fromkeys(CAMPI_TOKEN, 0)


def crea_clienti(cfg):
    """Un client per ogni provider usato in config.yaml. Chiavi ed endpoint vengono dal .env."""
    provider = {m["provider"] for m in cfg["modelli"]}
    clienti = {}
    if "openai" in provider:
        from openai import OpenAI
        clienti["openai"] = OpenAI()  # OPENAI_API_KEY, OPENAI_BASE_URL
    if "anthropic" in provider:
        from anthropic import AnthropicFoundry
        clienti["anthropic"] = AnthropicFoundry()  # ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_BASE_URL
    return clienti


def cache_attiva(cfg, provider):
    """True se per il provider si invia cache_control esplicito (config.yaml -> cache)."""
    return bool((cfg.get("cache") or {}).get(provider, False))


def chiama(m, clienti, messaggi, cfg, testo, etichette, cache=None):
    """cache=None usa config.yaml; True/False lo forza (serve al test di equivalenza)."""
    livello_ragionamento = cfg["livello_ragionamento"][m["provider"]]
    if m["provider"] == "openai":
        return chiama_openai(
            clienti["openai"], m["nome"], messaggi, cfg["temperatura"], livello_ragionamento
        )
    if m["provider"] == "anthropic":
        return chiama_anthropic(
            clienti["anthropic"], m["nome"], messaggi, cfg["temperatura"], livello_ragionamento,
            cache=cache_attiva(cfg, "anthropic") if cache is None else cache,
        )
    if m["provider"] == "mock":
        return chiama_mock(testo, etichette)
    raise SystemExit(f"Provider sconosciuto: {m['provider']}")


def riepiloga_token(records):
    """Somma i token registrati, mantenendo un totale anche per provider/modello."""
    totale_in = totale_out = 0
    per_modello = {}
    for record in records:
        token_in = int(record.get("token_in") or 0)
        token_out = int(record.get("token_out") or 0)
        totale_in += token_in
        totale_out += token_out
        chiave = (record.get("provider", "?"), record.get("modello", "sconosciuto"))
        modello_in, modello_out = per_modello.get(chiave, (0, 0))
        per_modello[chiave] = (modello_in + token_in, modello_out + token_out)
    return totale_in, totale_out, per_modello


# ---------------------------------------------------------------- validazione
def valida(grezzo, etichette):
    """Restituisce il dict se il JSON è valido e la categoria ammessa, altrimenti solleva ValueError."""
    try:
        out = json.loads(grezzo)
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON non valido: {e}")
    for k in ("motivazione", "quote", "categoria"):
        if k not in out:
            raise ValueError(f"chiave mancante: {k}")
    if not isinstance(out["quote"], str):
        raise ValueError("'quote' deve contenere una sola stringa")
    if out["categoria"] not in etichette:
        raise ValueError(f"categoria non ammessa: {out['categoria']!r}")
    return out


# ---------------------------------------------------------------- piano
def pianifica(cfg, storico):
    """Set, risposte, strategie, chiamate previste e quelle ancora da fare. Non chiama nessun modello."""
    set_esecuzione = "serbatoio" if cfg.get("calibrazione_serbatoio", False) else "valutazione"
    if set_esecuzione == "serbatoio":
        if "zero_shot" not in cfg["strategie"]:
            raise ValueError("La calibrazione sul serbatoio richiede la strategia zero_shot")
        strategie = ["zero_shot"]
        n_risposte = cfg["split"]["n_serbatoio"]
    else:
        strategie = cfg["strategie"]
        n_risposte = cfg["n_risposte"]
    risposte = pd.read_csv(
        SPLITS / f"{set_esecuzione}.csv", dtype={"id": str}, encoding="utf-8-sig"
    ).head(n_risposte)
    esperimento_id = cfg["esperimento_id"]
    fatte = {
        (r.get("esperimento_id", "legacy"), r.get("set", "valutazione"), r.get("provider"), r["modello"], r["strategia"], r["run"], str(r["id"]))
        for r in storico if r["valido"]
    }
    lavori = [
        (m, s, run, riga)
        for m in cfg["modelli"]
        for s in strategie
        for run in range(1, cfg["n_run"] + 1)
        for riga in risposte.itertuples()
        if (esperimento_id, set_esecuzione, m["provider"], m["nome"], s, run, str(riga.id)) not in fatte
    ]
    totale = len(cfg["modelli"]) * len(strategie) * cfg["n_run"] * len(risposte)
    return set_esecuzione, risposte, lavori, totale


# ---------------------------------------------------------------- main
def main():
    load_dotenv(ROOT / ".env", encoding="utf-8")
    cfg = carica_config()
    etichette = tutte_le_etichette(cfg)

    storico = leggi_calls()
    set_esecuzione, risposte, lavori, totale = pianifica(cfg, storico)
    esperimento_id = cfg["esperimento_id"]
    totale_token_in, totale_token_out, token_per_modello = riepiloga_token(storico)
    chiamate_esperimento = [
        r for r in storico if r.get("esperimento_id", "legacy") == esperimento_id
    ]
    token_esperimento_in, token_esperimento_out, token_per_modello_esperimento = riepiloga_token(
        chiamate_esperimento
    )
    modelli = ", ".join(f"{m['provider']}/{m['nome']}" for m in cfg["modelli"])
    strategie = ", ".join(cfg["strategie"])
    print(
        f"Esperimento: {esperimento_id} | set={set_esecuzione} "
        f"domanda={cfg['dati']['domanda_attiva']} seed={cfg['seed']} "
        f"risposte={len(risposte)} run={cfg['n_run']} "
        f"modelli=[{modelli}] strategie=[{strategie}] temperatura={cfg['temperatura']} "
        f"cache_claude={cache_attiva(cfg, 'anthropic')}"
    )
    print(f"Chiamate totali: {totale}  già fatte: {totale - len(lavori)}  da fare: {len(lavori)}")

    clienti = crea_clienti(cfg)

    CALLS.parent.mkdir(parents=True, exist_ok=True)
    n_errori = 0
    with open(CALLS, "a", encoding="utf-8") as f:
        for i, (m, strategia, run, riga) in enumerate(lavori, 1):
            messaggi = costruisci_messaggi(riga.testo, strategia, cfg)
            errori, out, grezzo, versione = [], None, None, None
            token = dict.fromkeys(CAMPI_TOKEN, 0)

            for tentativo in range(1, cfg["max_tentativi"] + 1):
                try:
                    grezzo, versione, t = chiama(m, clienti, messaggi, cfg, riga.testo, etichette)
                    token = {k: token[k] + t[k] for k in CAMPI_TOKEN}
                    out = valida(grezzo, etichette)
                    break
                except ValueError as e:          # output non valido: riprova subito
                    errori.append(str(e))
                except Exception as e:           # errore API (rete, rate limit...): attendi e riprova
                    errori.append(f"{type(e).__name__}: {e}")
                    time.sleep(2 ** tentativo)

            n_errori += len(errori)
            record = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "provider": m["provider"],
                "modello": m["nome"],
                "modello_versione": versione,
                "esperimento_id": esperimento_id,
                "set": set_esecuzione,
                "domanda": cfg["dati"]["domanda_attiva"],
                "seed": cfg["seed"],
                "n_risposte": len(risposte),
                "n_run": cfg["n_run"],
                "temperatura": cfg["temperatura"],
                "livello_ragionamento": cfg["livello_ragionamento"][m["provider"]],
                # cache_control esplicito inviato; per GPT la cache è automatica e si vede
                # solo da token_cache_lettura
                "cache_enabled": cache_attiva(cfg, m["provider"]),
                "strategia": strategia,
                "run": run,
                "id": str(riga.id),
                "valido": out is not None,
                "categoria": out["categoria"] if out else None,
                "motivazione": out["motivazione"] if out else None,
                "quote": out["quote"] if out else None,
                "tentativi": len(errori) + (1 if out else 0),
                "errori": errori,
                "output_grezzo": grezzo,
                **token,
            }
            tok_in, tok_out = token["token_in"], token["token_out"]
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            totale_token_in += tok_in
            totale_token_out += tok_out
            token_esperimento_in += tok_in
            token_esperimento_out += tok_out
            chiave_modello = (m["provider"], m["nome"])
            modello_in, modello_out = token_per_modello.get(chiave_modello, (0, 0))
            token_per_modello[chiave_modello] = (modello_in + tok_in, modello_out + tok_out)
            modello_in, modello_out = token_per_modello_esperimento.get(chiave_modello, (0, 0))
            token_per_modello_esperimento[chiave_modello] = (modello_in + tok_in, modello_out + tok_out)

            stato = record["categoria"] if out else "FALLITA"
            print(f"[{i}/{len(lavori)}] {m['nome']} {strategia} run={run} id={riga.id} -> {stato}")

    print(f"Fatto. Errori/ritentativi in questa sessione: {n_errori}")
    print("\nRiepilogo token:")
    print(
        f"Esperimento {esperimento_id}: input={token_esperimento_in:,} "
        f"output={token_esperimento_out:,} totale={token_esperimento_in + token_esperimento_out:,}"
    )
    for (provider, nome), (token_in, token_out) in sorted(token_per_modello_esperimento.items()):
        print(f"  {provider}/{nome}: input={token_in:,} output={token_out:,}")
    print(
        f"Totale storico: input={totale_token_in:,} output={totale_token_out:,} "
        f"totale={totale_token_in + totale_token_out:,}"
    )
    for (provider, nome), (token_in, token_out) in sorted(token_per_modello.items()):
        print(f"  {provider}/{nome}: input={token_in:,} output={token_out:,}")


if __name__ == "__main__":
    main()
