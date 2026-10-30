# Codifica qualitativa con LLM

Questo progetto usa due modelli linguistici (LLM) per codificare le risposte aperte di un questionario e
confronta il loro lavoro con quello di una codificatrice umana. Serve a chi fa analisi qualitativa e vuole capire
quanto ci si può fidare di un LLM come codificatore, e quali risposte conviene comunque far rivedere a una persona.

## Cosa fa

In un questionario le domande aperte ("Cosa ti è piaciuto di più?") producono testi liberi. Per analizzarli
qualcuno deve **codificarli**: leggere ogni risposta e assegnarle una categoria di un *codebook* (per esempio
`Social`, `Learning`, `Tool`). È un lavoro lento e in parte soggettivo.

La pipeline:

1. legge le risposte e la codifica umana da un file Excel;
2. chiede a ciascun modello di codificare ogni risposta, più volte (*run*). Il modello restituisce in JSON una
   categoria, una breve motivazione e una citazione letterale dal testo;
3. per ogni risposta sceglie l'etichetta della maggioranza delle run (**etichetta modale**);
4. confronta l'etichetta modale con la codifica umana (κ di Cohen, accordo percentuale, F1). Misura anche quanto
   il modello è stabile tra le run (κ di Fleiss) e se le citazioni compaiono davvero nel testo.

I modelli usati sono `gpt-5.6-luna` (Azure OpenAI) e `claude-haiku-5-5` (Claude su Azure AI Foundry). Le
strategie di prompt sono tre: *zero-shot* (solo il codebook), *one-shot* (un esempio per categoria) e
*few-shot* (fino a tre esempi per categoria).

## Struttura del repository

```
config.yaml                 unico punto di controllo: dati, codebook, modelli, strategie, esperimento
.env.example                modello per il file .env con chiavi ed endpoint delle API
prompts/system.txt          prompt di sistema ({codebook} viene sostituito dalle categorie di config.yaml)
prompts/esempi/             esempi one/few-shot scelti per ogni domanda
data/esempio/               Excel con risposte inventate, per provare la pipeline
data/raw/                   qui va l'Excel con i dati reali (non incluso)
scripts/01_split.py … 05_metrics.py   i passi della pipeline, da lanciare in ordine
scripts/common.py, metriche.py        funzioni condivise
scripts/analisi_revisione.py          quali risposte mandare a revisione umana
scripts/discordanti.py                risposte su cui le run non sono d'accordo
scripts/report_serbatoio.py           stabilità dopo una calibrazione sul serbatoio
tests/                      test automatici
```

Gli script scrivono split e risultati in `data/splits/` e `results/`, che non sono nel repository.

## Requisiti e installazione

Serve Python 3.11 o successivo.

```bash
git clone https://github.com/StefMila/LLMinQualitativeAnalysis.git
cd LLMinQualitativeAnalysis
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Per usare i modelli veri copia `.env.example` in `.env` e inserisci chiavi ed endpoint:

```bash
cp .env.example .env
```

Per l'esempio qui sotto il file `.env` non serve.

## Esempio minimo (senza API e senza costi)

Il provider `mock` risponde con categorie casuali. Serve a provare tutta la pipeline senza chiamare nessuna API:
l'accordo con la codifica umana sarà quindi vicino a zero.

1. In `config.yaml` usa i dati di esempio e il modello finto:

   ```yaml
   dati:
     excel: data/esempio/risposte_esempio.xlsx
   ...
   modelli:
     - provider: mock
       nome: gpt-5.6-luna
     - provider: mock
       nome: claude-haiku-5-5
   ```

2. Lancia i passi dalla cartella del progetto:

   ```bash
   python scripts/01_split.py       # divide le risposte in serbatoio (50), valutazione (300) e resto
   python scripts/01b_esempi.py     # sceglie gli esempi one/few-shot dal serbatoio
   python scripts/02_prompt.py      # mostra il prompt completo della prima risposta
   python scripts/03_run.py         # codifica: modello × strategia × run × risposta
   python scripts/04_aggregate.py   # etichetta modale e verifica delle citazioni
   python scripts/05_metrics.py     # metriche di accordo
   ```

3. Apri `results/piaciuto/metrics/riepilogo.csv`: c'è una riga per esperimento, modello e strategia.

`01b_esempi.py` riscrive `prompts/esempi/<domanda>/`. Per tornare agli esempi originali:
`git checkout prompts/esempi`.

## Uso con i modelli veri

1. Metti l'Excel in `data/raw/` e indica in `config.yaml` il foglio e le colonne (con la lettera di Excel) di
   ogni domanda. L'`id` di una risposta è il suo numero di riga.
2. In `config.yaml` scegli domanda (`dati.domanda_attiva`), modelli, strategie e livello di ragionamento, e assegna
   un **nuovo** `esperimento_id`. L'id viene salvato in ogni chiamata e tiene separati gli esperimenti. Le opzioni
   possibili sono scritte nei commenti del file.
3. Fai una prova breve (2 risposte, 1 run):

   ```bash
   python scripts/03_run.py --prova
   ```

4. Lancia `03_run.py`, poi `04_aggregate.py` e `05_metrics.py`.

`03_run.py` scrive ogni chiamata appena arriva. Se si interrompe, rilancialo: salta le chiamate già riuscite e
rifà quelle fallite. Con `n_thread` > 1 fa più chiamate in parallelo.

Per analizzare un'altra domanda cambia `dati.domanda_attiva` e riparti da `01_split.py`: split e risultati di ogni
domanda stanno in cartelle separate.

### Output

In `results/<domanda>/`:

- `calls.jsonl`: una riga per chiamata, con parametri, categoria, motivazione, citazione, errori e token.
  Il file cresce e non viene mai riscritto.
- `aggregated.csv`: etichetta modale e distribuzione delle run per ogni risposta.
- `metrics/riepilogo.csv`: tutte le metriche, una riga per esperimento, modello e strategia.
- `metrics/per_categoria.csv`, `metrics/stabilita.csv`, `metrics/confusione_*.csv`: metriche per categoria,
  per gruppo di stabilità e matrici di confusione.

### Regole principali

- **Etichetta modale**: una risposta la riceve solo se una categoria ha la maggioranza stretta delle run e nessuna
  run manca. Altrimenti resta senza etichetta.
- **Stabilità**: `stabile` = 10/10 run uguali, `abbastanza_stabile` = 8–9/10, `instabile` = 7/10 o meno.
- **Citazioni**: `esatta` se compare nel testo (a meno di maiuscole, spazi e apostrofi), `approssimata` se la
  somiglianza (`rapidfuzz.partial_ratio`) è almeno 90, altrimenti `non_trovata`.
- **Esempi**: si scelgono solo dal serbatoio, mai tra le risposte usate per la valutazione.

## Riprodurre i risultati

I risultati non sono nel repository perché contengono le risposte degli studenti. Per rigenerarli servono l'Excel
originale (in `data/raw/risposte.xlsx`, disponibile solo al gruppo di ricerca) e l'accesso alle API.

Ogni esperimento si ottiene impostando `config.yaml` come in tabella e lanciando i passi 03–05. Tutti usano 10 run
su 300 risposte di valutazione, salvo dove indicato.

| Esperimento (`esperimento_id`) | Domanda | Strategia | Ragionamento |
|---|---|---|---|
| `exp07_zeroShot_nM` | piaciuto | zero_shot | medium |
| `exp06_oneShot_nM` | piaciuto | one_shot | medium |
| `exp05_fewShot_nM` | piaciuto | few_shot | medium |
| `exp08_zeroShot_nM_high` | piaciuto | zero_shot | high |
| `exp11_fewShot_nM_high` | piaciuto | few_shot | high |
| `exp10_zeroShot_nM_high_q2` | descrizione | zero_shot | medium |
| `exp12_resto_zeroShot_nM` | piaciuto, set `resto` (778 risposte) | zero_shot | medium |

Seed fissi (`seed` ed `esempi.seed`) rendono uguali split ed esempi. Le risposte dei modelli possono comunque
variare da una chiamata all'altra: per questo ogni risposta è codificata 10 volte.

## Test

```bash
python -m unittest discover -s tests
```

## Licenza

MIT, vedi [LICENSE](LICENSE).
