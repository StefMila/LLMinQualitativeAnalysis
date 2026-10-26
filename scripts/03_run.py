"""Passo 3: chiama il modello per ogni combinazione modello x strategia x run x risposta.

Ogni chiamata viene aggiunta a results/<domanda>/calls.jsonl appena arriva. Rilanciando lo
script si saltano le combinazioni già riuscite (quelle fallite vengono ritentate).
Con n_thread > 1 (config.yaml) partono più chiamate insieme, sempre indipendenti tra loro.
"""
import argparse
import copy
import importlib
import json
import random
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from itertools import zip_longest

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


# Stesse chiavi che system.txt chiede a tutti i modelli; la categoria resta libera (niente enum)
# così nessun modello è vincolato più dell'altro, e valida() controlla che sia ammessa.
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


# Limite di output uguale per i due provider; comprende il ragionamento interno.
MAX_TOKEN_OUT = 16000


def chiama_openai(client, modello, messaggi, temperatura, livello_ragionamento):
    extra = {} if temperatura is None else {"temperature": temperatura}
    risp = client.chat.completions.create(
        model=modello,
        max_completion_tokens=MAX_TOKEN_OUT,  # come max_tokens di Claude: ragionamento compreso
        messages=messaggi,
        # stesso schema di Claude: entrambi i modelli sono vincolati alle stesse tre chiavi
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "codifica", "schema": SCHEMA_JSON, "strict": True},
        },
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
        max_tokens=MAX_TOKEN_OUT,  # include il ragionamento interno (thinking), non solo il JSON
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
    """Un client per ogni provider usato in config.yaml. Chiavi ed endpoint vengono dal .env.

    max_retries=0: gli SDK non ritentano da soli (lo farebbero anche sui 429, senza dirlo);
    tutti i ritentativi passano da esegui_chiamata, che li conta.
    """
    provider = {m["provider"] for m in cfg["modelli"]}
    clienti = {}
    if "openai" in provider:
        from openai import OpenAI
        clienti["openai"] = OpenAI(max_retries=0)  # OPENAI_API_KEY, OPENAI_BASE_URL
    if "anthropic" in provider:
        from anthropic import AnthropicFoundry
        # ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_BASE_URL
        clienti["anthropic"] = AnthropicFoundry(max_retries=0)
    return clienti


def cache_attiva(cfg, provider):
    """True se per il provider si invia cache_control esplicito (config.yaml -> cache)."""
    return bool((cfg.get("cache") or {}).get(provider, False))


def chiama(m, clienti, messaggi, cfg, testo, etichette, cache=None):
    """cache=None usa config.yaml; True/False lo forza (serve al test di equivalenza)."""
    livello_ragionamento = cfg["livello_ragionamento"].get(m["provider"])  # None per mock
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


def stampa_token(records, titolo):
    """Token di input e output sommati, in totale e per provider/modello."""
    per_modello = {}
    for r in records:
        chiave = (r.get("provider", "?"), r.get("modello", "sconosciuto"))
        t_in, t_out = per_modello.get(chiave, (0, 0))
        per_modello[chiave] = (t_in + int(r.get("token_in") or 0), t_out + int(r.get("token_out") or 0))
    totale_in = sum(t for t, _ in per_modello.values())
    totale_out = sum(t for _, t in per_modello.values())
    print(f"{titolo}: input={totale_in:,} output={totale_out:,} totale={totale_in + totale_out:,}")
    for (provider, nome), (t_in, t_out) in sorted(per_modello.items()):
        print(f"  {provider}/{nome}: input={t_in:,} output={t_out:,}")


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


# ---------------------------------------------------------------- ritentativi
# Tre tipi di errore, contati a parte in ogni record:
#   errori_formato   output non valido (JSON, chiavi, categoria): è una metrica del modello
#   errori_api       rete o servizio (5xx, timeout...): attesa 2^n e nuovo tentativo
#   ritentativi_429  quota del provider superata: dipende dal carico, non dal modello
# Formato e API condividono max_tentativi; i 429 hanno un budget a parte (max_tentativi_429)
# e non entrano né in "tentativi" né in "errori", così restano confrontabili con exp03.
_jitter = random.Random()  # generatore a parte: non tocca random usato dal mock


def e_429(e):
    return getattr(e, "status_code", None) == 429


def attesa_429(e, n):
    """Secondi da aspettare dopo l'n-esimo 429: Retry-After se c'è, altrimenti 2^n (max 60).

    Si aggiunge un'attesa casuale (fino a un quarto, almeno 1 s): i thread fermati dallo
    stesso 429 non ripartono tutti nello stesso istante.
    """
    intestazioni = getattr(getattr(e, "response", None), "headers", None) or {}
    attesa = None
    try:
        if intestazioni.get("retry-after-ms"):
            attesa = float(intestazioni["retry-after-ms"]) / 1000
        elif intestazioni.get("retry-after"):
            attesa = float(intestazioni["retry-after"])
    except (TypeError, ValueError):  # Retry-After come data HTTP: si usa il default
        attesa = None
    if attesa is None:
        attesa = min(2 ** n, 60)
    return attesa + _jitter.uniform(0, max(1.0, attesa / 4))


def esegui_chiamata(lavoro, clienti, cfg, etichette, contesto, stop=None):
    """Una chiamata con i suoi ritentativi; restituisce il record per calls.jsonl.

    contesto: esperimento_id, set, n_risposte, n_thread. stop (threading.Event) interrompe
    le attese quando lo script viene fermato; in quel caso restituisce None.
    """
    m, strategia, run, riga = lavoro
    stop = stop or threading.Event()
    messaggi = costruisci_messaggi(riga.testo, strategia, cfg)
    errori, out, grezzo, versione = [], None, None, None
    n_formato = n_api = n_429 = 0
    token = dict.fromkeys(CAMPI_TOKEN, 0)
    max_429 = cfg.get("max_tentativi_429", cfg["max_tentativi"])

    while n_formato + n_api < cfg["max_tentativi"] and n_429 <= max_429:
        if stop.is_set():
            return None
        try:
            grezzo, versione, t = chiama(m, clienti, messaggi, cfg, riga.testo, etichette)
            token = {k: token[k] + t[k] for k in CAMPI_TOKEN}
            out = valida(grezzo, etichette)
            break
        except ValueError as e:          # output non valido: riprova subito
            n_formato += 1
            errori.append(str(e))
        except Exception as e:
            if e_429(e):                 # quota: attendi quanto chiede il provider e riprova
                n_429 += 1
                if n_429 <= max_429:
                    stop.wait(attesa_429(e, n_429))
            else:                        # errore API (rete, servizio...): attendi e riprova
                n_api += 1
                errori.append(f"{type(e).__name__}: {e}")
                stop.wait(2 ** (n_formato + n_api))
    if stop.is_set() and out is None:
        return None

    return {
        "ts": datetime.now(timezone.utc).isoformat(),
        "provider": m["provider"],
        "modello": m["nome"],
        "modello_versione": versione,
        "esperimento_id": contesto["esperimento_id"],
        "set": contesto["set"],
        "domanda": cfg["dati"]["domanda_attiva"],
        "seed": cfg["seed"],
        "n_risposte": contesto["n_risposte"],
        "n_run": cfg["n_run"],
        "n_thread": contesto["n_thread"],
        "temperatura": cfg["temperatura"],
        "livello_ragionamento": cfg["livello_ragionamento"].get(m["provider"]),
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
        "tentativi": n_formato + n_api + (1 if out else 0),  # senza i 429, come in exp03
        "errori": errori,
        "errori_formato": n_formato,
        "errori_api": n_api,
        "ritentativi_429": n_429,
        "output_grezzo": grezzo,
        **token,
    }


# ---------------------------------------------------------------- esecuzione
def alterna_modelli(lavori):
    """Alterna i lavori dei modelli (A1, B1, A2, B2...): con più thread i provider lavorano insieme."""
    per_modello = {}
    for lavoro in lavori:
        per_modello.setdefault((lavoro[0]["provider"], lavoro[0]["nome"]), []).append(lavoro)
    return [l for gruppo in zip_longest(*per_modello.values()) for l in gruppo if l is not None]


def esegui_lavori(lavori, clienti, cfg, etichette, contesto, percorso, al_record=None):
    """Esegue i lavori con n_thread chiamate insieme e scrive ogni record appena arriva.

    Le chiamate restano indipendenti e senza cronologia: cambia solo l'ordine di arrivo.
    Scrive solo il thread principale, una riga intera alla volta. Con Ctrl+C o un errore
    le attese si interrompono, i lavori non partiti si annullano e le chiamate in volo si
    scartano: non compaiono in calls.jsonl e al rilancio vengono rifatte.
    """
    n_thread = max(1, int(cfg.get("n_thread", 1)))
    if n_thread > 1:
        lavori = alterna_modelli(lavori)
    stop = threading.Event()
    percorso.parent.mkdir(parents=True, exist_ok=True)
    # se l'ultima riga è rimasta a metà (processo ucciso mentre scriveva), si va a capo:
    # leggi_calls la salta e la nuova riga non ci si attacca
    a_capo = percorso.exists() and percorso.stat().st_size > 0 and percorso.read_bytes()[-1:] != b"\n"

    pool = ThreadPoolExecutor(max_workers=n_thread)
    try:
        with open(percorso, "a", encoding="utf-8") as f:
            if a_capo:
                f.write("\n")
            in_corso = {
                pool.submit(esegui_chiamata, l, clienti, cfg, etichette, contesto, stop) for l in lavori
            }
            while in_corso:
                # timeout breve: su Windows un'attesa senza timeout non sente Ctrl+C
                finiti, in_corso = wait(in_corso, timeout=1, return_when=FIRST_COMPLETED)
                for futuro in finiti:
                    record = futuro.result()
                    if record is None:
                        continue
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    if al_record:
                        al_record(record)
    except BaseException:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


# ---------------------------------------------------------------- main
def applica_prova(cfg):
    """--prova: 2 risposte e 1 run con id <esperimento_id>_prova; il resto del config resta uguale."""
    return cfg | {"n_risposte": 2, "n_run": 1, "esperimento_id": f"{cfg['esperimento_id']}_prova"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prova", action="store_true",
                        help="prova breve (2 risposte, 1 run) con id <esperimento_id>_prova")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env", encoding="utf-8")
    cfg = carica_config()
    if args.prova:
        cfg = applica_prova(cfg)
    etichette = tutte_le_etichette(cfg)

    set_esecuzione, risposte, lavori, totale = pianifica(cfg, leggi_calls())
    esperimento_id = cfg["esperimento_id"]
    n_thread = max(1, int(cfg.get("n_thread", 1)))
    modelli = ", ".join(f"{m['provider']}/{m['nome']}" for m in cfg["modelli"])
    print(
        f"Esperimento: {esperimento_id} | set={set_esecuzione} "
        f"domanda={cfg['dati']['domanda_attiva']} seed={cfg['seed']} "
        f"risposte={len(risposte)} run={cfg['n_run']} "
        f"modelli=[{modelli}] strategie=[{', '.join(cfg['strategie'])}] temperatura={cfg['temperatura']} "
        f"cache_claude={cache_attiva(cfg, 'anthropic')} n_thread={n_thread}"
    )
    print(f"Chiamate totali: {totale}  già fatte: {totale - len(lavori)}  da fare: {len(lavori)}")

    contesto = {"esperimento_id": esperimento_id, "set": set_esecuzione,
                "n_risposte": len(risposte), "n_thread": n_thread}
    sessione = {"fatte": 0, "fallite": 0, "errori_formato": 0, "errori_api": 0, "ritentativi_429": 0}

    def al_record(record):
        sessione["fatte"] += 1
        sessione["fallite"] += not record["valido"]
        for k in ("errori_formato", "errori_api", "ritentativi_429"):
            sessione[k] += record[k]
        stato = record["categoria"] if record["valido"] else "FALLITA"
        print(
            f"[{sessione['fatte']}/{len(lavori)}] {record['modello']} {record['strategia']} "
            f"run={record['run']} id={record['id']} -> {stato}"
        )

    esegui_lavori(lavori, crea_clienti(cfg), cfg, etichette, contesto, CALLS, al_record)

    print(
        f"Fatto. Chiamate fallite: {sessione['fallite']} | errori di formato: "
        f"{sessione['errori_formato']} | errori API: {sessione['errori_api']} | "
        f"ritentativi per 429 (quota): {sessione['ritentativi_429']}"
    )
    print("\nRiepilogo token:")
    storico = leggi_calls()
    stampa_token([r for r in storico if r.get("esperimento_id", "legacy") == esperimento_id],
                 f"Esperimento {esperimento_id}")
    stampa_token(storico, "Totale storico")


if __name__ == "__main__":
    main()
