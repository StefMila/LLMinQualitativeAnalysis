# codifica_llm

Codifica qualitativa di risposte aperte con modelli linguistici (LLM), confrontata con la codifica umana.

## Il problema

Nei questionari le domande aperte ("Cosa ti è piaciuto di più?") danno risposte libere che, per essere analizzate,
vanno **codificate**: un codificatore umano legge ogni risposta e le assegna una categoria di un *codebook*
(es. `Social`, `Learning`, `Tool`). È un lavoro lento e in parte soggettivo.

Questo progetto misura quanto un LLM si avvicina a una codificatrice umana sullo stesso compito. I dati sono le
risposte di studenti a un questionario dopo un laboratorio scolastico (tecnologie, robotica, programmazione, scienze).
Il modello riceve il codebook, la domanda e una risposta, e restituisce in JSON una categoria, una breve motivazione
e una citazione letterale della risposta che giustifica la scelta.

Il confronto riguarda:

- **due modelli**: `gpt-5.6-luna` (Azure OpenAI) e `claude-haiku-5-5` (Azure AI Foundry);
- **tre strategie di prompt**: *zero-shot* (solo codebook), *one-shot* (un esempio per categoria) e
  *few-shot* (fino a tre esempi per categoria);
- **due livelli di ragionamento** dei modelli (`medium` e `high`).

Ogni risposta è codificata 10 volte (*run*) per misurare la stabilità del modello. L'etichetta finale è quella della
maggioranza delle run (**etichetta modale**) e viene confrontata con la prima codifica umana (`Coding 1`) tramite
κ di Cohen, accordo percentuale e F1. Si misurano anche la stabilità tra le run (κ di Fleiss, TA@10), gli errori di
formato e quante citazioni si ritrovano davvero nel testo.

## Struttura

```
config.yaml          unico punto di controllo: dati, codebook, modelli, strategia, esperimento
prompts/system.txt   prompt di sistema ({codebook} viene sostituito dalle categorie di config.yaml)
prompts/esempi/      esempi one/few-shot scelti per ogni domanda e un prompt di esempio
data/raw/            Excel con risposte e codifica umana
data/splits/         serbatoio / valutazione / resto per ogni domanda
scripts/             pipeline (01–05) e script di analisi
results/<domanda>/   log delle chiamate, aggregazioni e metriche
tests/               test automatici
```

## Installazione

Serve Python 3.11 o successivo.

```bash
python -m venv .venv
source .venv/Scripts/activate      # Windows (Git Bash); su Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env               # poi inserisci in .env chiavi ed endpoint
```

Per provare la pipeline senza API e senza costi, in `config.yaml` metti `provider: mock` a entrambi i modelli:
il modello finto risponde con categorie casuali, quindi l'accordo sarà vicino a zero.

## Uso

Gli script si lanciano dalla cartella del progetto, uno dopo l'altro. Ognuno legge l'output del precedente:
se qualcosa va storto si rilancia solo quel passo.

| Passo | Script | Cosa fa | Scrive |
|---|---|---|---|
| 1 | `python scripts/01_split.py` | legge l'Excel, scarta le risposte vuote o non codificate, mescola con seed fisso e divide in serbatoio (50), valutazione (300) e resto | `data/splits/<domanda>/` |
| 1b | `python scripts/01b_esempi.py` | sceglie gli esempi one/few-shot dal serbatoio con una regola fissa (vedi sotto) | `prompts/esempi/<domanda>/` |
| 2 | `python scripts/02_prompt.py` | mostra il prompt completo per la prima risposta, per controllarlo | `prompt_esempio_<strategia>.txt` |
| 3 | `python scripts/03_run.py` | chiama i modelli: ogni modello × strategia × run × risposta | `results/<domanda>/calls.jsonl` |
| 4 | `python scripts/04_aggregate.py` | etichetta modale per risposta e verifica delle citazioni | `aggregated.csv`, `citazioni*.csv` |
| 5 | `python scripts/05_metrics.py` | metriche di accordo per esperimento, modello e strategia | `results/<domanda>/metrics/` |

### Un nuovo esperimento

1. In `config.yaml` scegli domanda (`dati.domanda_attiva`), modelli, strategia (`strategie`) e livello di
   ragionamento, e assegna un **nuovo** `esperimento_id`. L'id viene salvato in ogni chiamata e tiene separati
   i risultati degli esperimenti.
2. Fai una prova breve: `python scripts/03_run.py --prova` (2 risposte, 1 run, id `<esperimento_id>_prova`).
3. Lancia `03_run.py`, poi `04_aggregate.py` e `05_metrics.py`.

`03_run.py` scrive ogni chiamata appena arriva. Se si interrompe, rilancialo: salta le chiamate già riuscite e
rifà quelle fallite. Con `n_thread` > 1 fa più chiamate in parallelo, sempre indipendenti tra loro.

### Un'altra domanda

Cambia `dati.domanda_attiva` (`descrizione`, `piaciuto`, `non_piaciuto`, `diversamente`) e rilancia da
`01_split.py`. Split, esempi e risultati di ogni domanda stanno in cartelle separate e non si mescolano.

### Script di analisi

Non chiamano i modelli e lavorano sull'esperimento indicato in `config.yaml`.

| Script | Cosa fa |
|---|---|
| `analisi_revisione.py` | confronta regole per scegliere quali risposte far rivedere a una persona (es. "instabile per almeno un modello oppure modelli in disaccordo") con una revisione casuale della stessa quantità |
| `discordanti.py` | elenca le risposte in cui le run danno categorie diverse, con motivazione e citazione di ogni run |
| `report_serbatoio.py` | stabilità sul serbatoio dopo una calibrazione (`calibrazione_serbatoio: true`) |
| `prova_cache.py` | verifica che la prompt caching di Claude non cambi le risposte (`verifica`, `equivalenza` e `analisi`; le prime due chiamano l'API) |

### Test

```bash
python -m unittest discover -s tests
```

## Scelte di metodo

- **Dati.** Le colonne dell'Excel si indicano con la lettera (`col_testo`, `col_umano`) e l'`id` di una risposta
  è il suo numero di riga. Il riferimento è `Coding 1`: `Coding 2` e `Coding 3` sono codici aggiuntivi della stessa
  codificatrice (fino a tre per risposta) e non entrano nel confronto. In `Coding 1` "-" vuol dire non ancora
  codificata (risposta esclusa) e la cella vuota vuol dire `non classificabile`.
- **Esempi.** Si scelgono solo dal serbatoio, escludendo i testi identici a una risposta di valutazione. Per ogni
  categoria si mescolano le risposte con `esempi.seed`: il one-shot prende la prima, il few-shot le prime tre.
  Gli esempi sono un blocco fisso prima della risposta da codificare, uguale in ogni chiamata.
- **Chiamate.** Ogni chiamata è indipendente e non ha cronologia. I due modelli ricevono lo stesso prompt, lo stesso
  schema JSON e lo stesso limite di token. Un output non valido (JSON errato o categoria inesistente) viene
  ritentato e conta come errore di formato. Gli errori di quota (429) sono contati a parte.
- **Etichetta modale.** Una risposta riceve l'etichetta solo se una categoria ha la maggioranza stretta delle run
  previste e nessuna run manca. Altrimenti resta senza etichetta (`da_revisionare`).
- **Stabilità.** `stabile` = 10/10 run uguali, `abbastanza_stabile` = 8–9/10, `instabile` = 7/10 o meno, oppure
  nessuna etichetta modale.
- **Tre versioni delle metriche.** `principale` usa solo le risposte con etichetta modale e riporta la copertura;
  `controllo` usa tutte le risposte e conta come errore quelle senza etichetta; `confronto_modelli` usa solo le
  risposte con etichetta per tutti i modelli. Il F1 macro considera solo le categorie presenti nella codifica umana.
- **Citazioni.** Una citazione è `esatta` se compare nel testo dopo aver uniformato maiuscole, spazi, apostrofi e
  virgolette. Altrimenti si confronta per somiglianza (`rapidfuzz.partial_ratio`): è `approssimata` con un punteggio
  di almeno 90, `non_trovata` sotto.

## Output principali

In `results/<domanda>/`:

- `calls.jsonl`: una riga per chiamata, con esperimento, modello, parametri, categoria, motivazione, citazione,
  errori e token (input, output, cache, ragionamento). Il file si aggiunge e non si riscrive mai. Le righe dei
  primi esperimenti hanno meno campi e restano leggibili.
- `aggregated.csv`: una riga per esperimento, modello, strategia e risposta, con l'etichetta modale e la
  distribuzione delle run.
- `metrics/riepilogo.csv`: tutte le metriche, una riga per esperimento, modello e strategia.
  `metrics/tabella_tesi.csv` ne riporta le colonne principali.
- `metrics/per_categoria.csv`, `metrics/stabilita.csv` e `metrics/confusione_*.csv`: metriche per categoria e per
  gruppo di stabilità, e matrici di confusione.

## Dati

Le risposte sono testi liberi scritti da studenti e possono contenere nomi propri. Non vanno diffuse fuori dal
gruppo di ricerca, e lo stesso vale per i file in `results/` che riportano i testi.
