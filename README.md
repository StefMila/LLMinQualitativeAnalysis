# codifica_llm

Codifica di risposte aperte con LLM e confronto con la codifica umana.
Versione attuale: **prototipo Fase 1** — 1 modello (Azure OpenAI), zero-shot, 1 domanda, 20 risposte, 3 run.

## Installazione (una volta sola)

```bash
# Esegui dalla cartella del progetto
python -m venv --clear .venv
source .venv/Scripts/activate
pip install -r requirements.txt
cp .env.example .env        # poi apri .env e incolla la chiave Azure
```

Non copiare `.venv` da un altro PC: contiene riferimenti all'interprete Python e al percorso locale. Ricreala con il comando sopra.

## Dati e domande

- L'Excel è `data/raw/risposte.xlsx`, foglio `QL Coding`: per ogni domanda aperta c'è la colonna del testo seguita da `Coding 1/2/3`.
- Si confronta il modello solo con la **prima codifica** (`Coding 1`). Le colonne successive approfondiscono e per ora si ignorano.
- Nella prima codifica `-` = non ancora codificata (riga esclusa), cella vuota = `non classificabile`.
- Si analizza **una domanda alla volta**: in `config.yaml` cambia `dati.domanda_attiva`
  (`descrizione`, `piaciuto`, `non_piaciuto`, `diversamente`) e rilancia da `01_split.py`.
  Split e risultati finiscono in `data/splits/<domanda>/` e `results/<domanda>/`, quindi non si mescolano.
- Le colonne si indicano con la lettera di Excel; l'`id` è il numero di riga in Excel.
- Il codebook (nomi e definizioni) sta in `config.yaml` → `categorie` e viene inserito nel prompt al posto di `{codebook}` in `prompts/system.txt`.

## Pipeline

Ogni script legge l'output del precedente e scrive un file. Se qualcosa va storto, rilanci solo quel passo.
Si lanciano dalla cartella del progetto, ad es. `python scripts/01_split.py`.

| Script | Cosa fa | Scrive |
|---|---|---|
| `00_ciao.py` | chiamata di prova all'API | — |
| `01_split.py` | prende la domanda attiva, mescola con seed fisso e divide in serbatoio / valutazione / resto | `data/splits/<domanda>/*.csv` |
| `01b_esempi.py` | sceglie gli esempi one/few-shot dal serbatoio con regola fissa e `esempi.seed` (esclusi i testi uguali a risposte di valutazione) | `prompts/esempi/<domanda>/*.csv` |
| `02_prompt.py` | costruisce i messaggi; lanciato da solo stampa il prompt della prima risposta | — |
| `03_run.py` | chiama i modelli con temperatura e livello di ragionamento espliciti, ritenta JSON invalidi e mostra i token finali | `results/<domanda>/calls.jsonl` |
| `04_aggregate.py` | etichetta/quota modale, verifica citazioni, revisioni e sensibilità | `aggregated.csv`, `citazioni.csv`, report citazioni |
| `05_metrics.py` | κ di Cohen (con po e pe) e Fleiss, TA@R, accordo percentuale, F1 e tassi citazioni; `tabella_tesi.csv` | `results/<domanda>/metrics/` |
| `leggi.py` | stampa N output a caso per leggerli a mano (`python scripts/leggi.py 20`) | — |
| `checkQuotaModale.py` | mostra le risposte con categorie diverse tra le run dell'esperimento attivo | — |
| `tests/test_citation_verification.py` | test automatici delle citazioni e separazione dei set | `python -m unittest discover -s tests -v` |

Dopo la pipeline, esegui `python scripts/checkQuotaModale.py` per approfondire le risposte discordanti.

Il prompt restituisce una sola `quote` (stringa) per risposta; le vecchie chiamate con liste vengono
mantenute leggibili come storico. `verifica_citazioni` in `config.yaml` contiene la versione, la soglia e
le soglie descrittive della verifica. `citazioni.csv` ha una riga per citazione e run valida e salva set,
modello, strategia, run, punteggio, esito, versione e tratto più simile del testo raw. La verifica è esatta
dopo `ftfy` e normalizzazione Unicode, maiuscole, spazi, apostrofi e virgolette; altrimenti usa
`rapidfuzz.partial_ratio`: con soglia provvisoria 90 è `approssimata` sopra soglia, `non_trovata` sotto.
`05_metrics.py` calcola entrambi i tassi sulla valutazione,
su tutte le citazioni delle run valide per modello e strategia.
`citazioni_da_esaminare.csv` e `citazioni_da_esaminare_serbatoio.csv` contengono le citazioni non esatte,
con `giudizio_umano` vuota. `metrics/sensibilita_citazioni.csv` contiene conteggi per set e soglia;
è descrittivo e non sceglie la soglia.
Nel `riepilogo.csv`, `quota_risposte_con_tutte_le_citazioni_verificate` resta la metrica secondaria;
`ta_at_n` è la quota di risposte per cui tutte le R run valide concordano (TA@R). I log storici hanno
3 run; `config.yaml` è impostato a 10 per le prossime esecuzioni.
Una risposta con parità, senza maggioranza stretta o con run valide mancanti non riceve un'etichetta modale:
`etichetta_modale` è vuota e `da_revisionare` è vero.
κ di Cohen (con po e pe), accordo percentuale e F1 macro (complessivi e per categoria) sono riportati in tre versioni
(`scripts/metriche.py`, legenda in `metrics/versioni.csv`): `principale` solo risposte con etichetta modale, con la
copertura; `controllo` tutte le risposte, quelle senza etichetta contate come errore (nel F1 abbassano solo il
richiamo della categoria umana); `confronto_modelli` solo le risposte con etichetta modale per tutti i modelli.
In `riepilogo.csv` le colonne hanno il suffisso della versione, negli altri file c'è la colonna `versione`.
`metrics/stabilita.csv` riporta l'accordo con l'umano per gruppo di stabilità, definiti una sola volta in
`scripts/common.py` (`GRUPPI_STABILITA`) sulla quota modale rispetto alle run previste: `stabile` 10/10,
`abbastanza_stabile` 8–9/10, `instabile` 7/10 o meno, comprese le risposte senza etichetta modale.
`python scripts/report_serbatoio.py` descrive gli stessi gruppi sul pilota del serbatoio
(`serbatoio_stabilita.csv`), che non entra nelle metriche. κ di Fleiss usa solo le risposte con tutte le run valide; non misura
l'accordo su una singola risposta, che è descritto da `quota_modale`. `per_categoria.csv` contiene precision,
recall e F1 per categoria. Il F1 macro include solo le categorie con almeno un caso umano nel campione;
`non classificabile` resta nella matrice di confusione, e una sua predizione contro un'etichetta umana
diversa è un errore.

Per avviare la calibrazione sul serbatoio, imposta `calibrazione_serbatoio: true` in `config.yaml` e lancia
`python scripts/03_run.py`. Il runner usa solo zero-shot, tutte le 50 risposte del serbatoio e il numero di
run configurato; i record sono marcati `set: serbatoio`. Poi esegui `04_aggregate.py` per generare la
sensibilità e `citazioni_da_esaminare_serbatoio.csv`. La fase non parte automaticamente e non sceglie la
soglia: dopo l'esecuzione, riporta `calibrazione_serbatoio: false` prima delle run di valutazione.

`03_run.py` scrive ogni chiamata appena arriva: se si interrompe, rilancialo e salta quelle già riuscite
(le fallite vengono ritentate). Per ricominciare da zero, cancella `results/<domanda>/calls.jsonl`.

Ogni configurazione sperimentale ha un `esperimento_id` in `config.yaml`. Assegna un ID nuovo quando cambi
domanda, seed, numero di risposte o run, modelli, strategia, temperatura o livello di ragionamento. Il livello
è fissato per provider in `livello_ragionamento` (`openai: medium`, `anthropic: high`) e viene salvato in ogni chiamata.
Lo script salva l'ID in ogni record
e lo usa per distinguere resume, aggregazioni e metriche. I record precedenti senza ID sono mantenuti come
`legacy`. All'avvio `03_run.py` stampa i parametri; al termine mostra il riepilogo dei token
dell'esperimento e dello storico complessivo, suddiviso per modello.

## Provare senza spendere

In `config.yaml` metti `provider: mock`: il "modello" risponde a caso, senza API.
Serve solo a vedere che la pipeline gira dall'inizio alla fine (il κ sarà ~0).

## Note

- `calls.jsonl` salva per ogni riga l'`esperimento_id`, la versione esatta del modello (`modello_versione`), i parametri essenziali
  della baseline e i token input/output. I token dei tentativi con risposta sono sommati nella riga; il totale
  storico include tutte le righe, anche quelle di chiamate fallite e poi ritentate.
- Da exp04 ogni riga ha anche `cache_enabled` (cache_control esplicito inviato: `config.yaml` → `cache`),
  `token_cache_scrittura`, `token_cache_lettura` e `token_ragionamento`. `token_in` è sempre tutto l'input,
  in cache o no (per Claude = `input_tokens` + scrittura + lettura), e `token_out` comprende il ragionamento,
  come nei log precedenti. exp03 è stato eseguito senza cache, exp04 con cache per Claude.
- `python scripts/prova_cache.py verifica|equivalenza|analisi`: prove della cache sul serbatoio, con log separato
  in `results/<domanda>/prova_cache/` (non entra in aggregazioni e metriche).
- Con `provider: mock` il conteggio token è sempre zero: per misurare il consumo reale usa un provider API.
- Alcuni modelli di ragionamento non accettano `temperature`: in quel caso metti `temperatura: null` in `config.yaml`.
- Gli split dipendono dal `seed` e da `n_serbatoio` / `n_valutazione`: se li cambi, rilancia `01_split.py` e cancella i risultati vecchi.
