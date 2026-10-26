import copy
import json
import sys
import tempfile
import threading
import time
import unittest
from importlib import import_module
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

run = import_module("03_run")
common = import_module("common")

MODELLI = [{"provider": "openai", "nome": "gpt-finto"}, {"provider": "anthropic", "nome": "claude-finto"}]


def config(n_thread=1, **altro):
    cfg = copy.deepcopy(common.carica_config())
    cfg.update(
        modelli=MODELLI, strategie=["zero_shot"], n_risposte=4, n_run=2, n_thread=n_thread,
        esperimento_id="test_thread", calibrazione_serbatoio=False, max_tentativi=3,
        max_tentativi_429=8,
    )
    cfg.update(altro)
    return cfg


def risposte(n=4):
    return pd.DataFrame({"id": [str(i) for i in range(1, n + 1)], "testo": [f"risposta {i}" for i in range(1, n + 1)]})


def contesto(cfg):
    return {"esperimento_id": cfg["esperimento_id"], "set": "valutazione", "n_risposte": 4,
            "n_thread": cfg["n_thread"]}


def lavori(cfg):
    return [(m, "zero_shot", r, riga) for m in cfg["modelli"] for r in range(1, cfg["n_run"] + 1)
            for riga in risposte().itertuples()]


def token(n=1):
    return dict.fromkeys(run.CAMPI_TOKEN, n)


def ok(categoria="Teacher"):
    return json.dumps({"motivazione": "m", "quote": "q", "categoria": categoria}), "finto", token()


def chiave(r):
    return tuple(r[k] for k in common.CHIAVE)


class Errore429(Exception):
    def __init__(self, intestazioni=None):
        super().__init__("Error code: 429 - rate limit")
        self.status_code = 429
        self.response = SimpleNamespace(headers=intestazioni or {})


class Errore500(Exception):
    status_code = 500


class EsecuzioneParallelaTests(unittest.TestCase):
    def esegui(self, n_thread, percorso):
        cfg = config(n_thread)

        def chiama_lenta(*_args, **_kwargs):
            time.sleep(0.005)
            return ok()

        with mock.patch.object(run, "chiama", side_effect=chiama_lenta):
            run.esegui_lavori(lavori(cfg), {}, cfg, common.tutte_le_etichette(cfg), contesto(cfg), percorso)
        righe = percorso.read_text(encoding="utf-8").splitlines()
        return [json.loads(r) for r in righe]  # fallisce se una riga non è JSON completo

    def test_uno_e_otto_thread_producono_gli_stessi_record(self):
        with tempfile.TemporaryDirectory() as cartella:
            uno = self.esegui(1, Path(cartella) / "uno.jsonl")
            otto = self.esegui(8, Path(cartella) / "otto.jsonl")
        self.assertEqual(len(uno), 16)
        self.assertEqual(len(otto), 16)
        self.assertEqual(len({chiave(r) for r in otto}), 16)  # nessun doppione
        self.assertEqual({chiave(r) for r in uno}, {chiave(r) for r in otto})
        campi_variabili = {"ts", "n_thread"}
        per_chiave = {chiave(r): r for r in uno}
        for r in otto:
            atteso = {k: v for k, v in per_chiave[chiave(r)].items() if k not in campi_variabili}
            self.assertEqual({k: v for k, v in r.items() if k not in campi_variabili}, atteso)
        self.assertEqual({r["n_thread"] for r in otto}, {8})

    def test_con_piu_thread_i_modelli_si_alternano(self):
        cfg = config(8)
        ordine = [l[0]["nome"] for l in run.alterna_modelli(lavori(cfg))]
        self.assertEqual(ordine[:4], ["gpt-finto", "claude-finto", "gpt-finto", "claude-finto"])
        self.assertEqual(len(ordine), 16)


class RitentativiTests(unittest.TestCase):
    def esegui_una(self, effetti, cfg=None):
        cfg = cfg or config()
        lavoro = lavori(cfg)[0]
        with mock.patch.object(run, "chiama", side_effect=effetti), \
                mock.patch.object(run, "attesa_429", return_value=0):
            return run.esegui_chiamata(lavoro, {}, cfg, common.tutte_le_etichette(cfg), contesto(cfg))

    def test_errore_di_formato_contato_come_formato(self):
        r = self.esegui_una([ok(categoria=""), ok()])
        self.assertTrue(r["valido"])
        self.assertEqual((r["errori_formato"], r["errori_api"], r["ritentativi_429"]), (1, 0, 0))
        self.assertEqual(r["tentativi"], 2)
        self.assertEqual(len(r["errori"]), 1)

    def test_429_contati_a_parte_e_fuori_da_tentativi_ed_errori(self):
        r = self.esegui_una([Errore429(), Errore429(), ok()])
        self.assertTrue(r["valido"])
        self.assertEqual((r["errori_formato"], r["errori_api"], r["ritentativi_429"]), (0, 0, 2))
        self.assertEqual(r["tentativi"], 1)
        self.assertEqual(r["errori"], [])

    def test_429_hanno_piu_tentativi_degli_altri_errori(self):
        r = self.esegui_una([Errore429()] * 6 + [ok()])  # 6 > max_tentativi = 3
        self.assertTrue(r["valido"])
        self.assertEqual(r["ritentativi_429"], 6)

    def test_429_oltre_il_budget_la_chiamata_fallisce(self):
        cfg = config(max_tentativi_429=2)
        chiamate = [Errore429()] * 3 + [ok()]
        r = self.esegui_una(chiamate, cfg)
        self.assertFalse(r["valido"])
        self.assertEqual(r["ritentativi_429"], 3)
        self.assertEqual(r["errori"], [])

    def test_errori_api_usano_il_budget_normale(self):
        with mock.patch.object(run.threading.Event, "wait", return_value=False):
            r = self.esegui_una([Errore500(), Errore500(), Errore500(), ok()])
        self.assertFalse(r["valido"])
        self.assertEqual((r["errori_formato"], r["errori_api"], r["ritentativi_429"]), (0, 3, 0))


def errori_rete_e_servizio(lib):
    """Le eccezioni vere dell'SDK per timeout, connessione interrotta e 500/502/503/529."""
    import httpx2

    richiesta = httpx2.Request("POST", "https://esempio")
    errori = {
        "timeout": lib.APITimeoutError(request=richiesta),
        "connessione": lib.APIConnectionError(request=richiesta),
    }
    for codice in (500, 502, 503, 529):
        risposta = httpx2.Response(codice, request=richiesta)
        errori[str(codice)] = lib.APIStatusError(f"errore {codice}", response=risposta, body=None)
    return errori


class ErroriApiTests(unittest.TestCase):
    def test_rete_e_servizio_finiscono_in_errori_api_con_max_tentativi(self):
        import anthropic
        import openai

        cfg = config()
        for lib in (openai, anthropic):
            for nome, errore in errori_rete_e_servizio(lib).items():
                with self.subTest(sdk=lib.__name__, errore=nome):
                    effetti = [errore] * cfg["max_tentativi"] + [ok()]
                    with mock.patch.object(run, "chiama", side_effect=effetti) as chiama, \
                            mock.patch.object(run.threading.Event, "wait", return_value=False):
                        r = run.esegui_chiamata(
                            lavori(cfg)[0], {}, cfg, common.tutte_le_etichette(cfg), contesto(cfg)
                        )
                    self.assertEqual(chiama.call_count, cfg["max_tentativi"])  # ritentata, poi basta
                    self.assertFalse(r["valido"])
                    self.assertEqual((r["errori_formato"], r["errori_api"], r["ritentativi_429"]), (0, 3, 0))
                    self.assertEqual(len(r["errori"]), 3)

    def test_errore_api_poi_successo(self):
        import anthropic

        errore = errori_rete_e_servizio(anthropic)["529"]
        cfg = config()
        with mock.patch.object(run, "chiama", side_effect=[errore, ok()]), \
                mock.patch.object(run.threading.Event, "wait", return_value=False):
            r = run.esegui_chiamata(lavori(cfg)[0], {}, cfg, common.tutte_le_etichette(cfg), contesto(cfg))
        self.assertTrue(r["valido"])
        self.assertEqual((r["errori_formato"], r["errori_api"], r["tentativi"]), (0, 1, 2))


class Attesa429Tests(unittest.TestCase):
    def test_usa_retry_after_in_secondi_piu_jitter(self):
        attese = [run.attesa_429(Errore429({"retry-after": "20"}), 1) for _ in range(50)]
        self.assertTrue(all(20 <= a <= 25 for a in attese))
        self.assertGreater(len(set(attese)), 1)  # il jitter cambia l'attesa

    def test_usa_retry_after_ms(self):
        a = run.attesa_429(Errore429({"retry-after-ms": "1500"}), 1)
        self.assertTrue(1.5 <= a <= 2.5)

    def test_senza_retry_after_attesa_esponenziale_con_tetto(self):
        self.assertTrue(4 <= run.attesa_429(Errore429(), 2) <= 5)
        self.assertTrue(60 <= run.attesa_429(Errore429(), 10) <= 75)

    def test_retry_after_come_data_usa_il_default(self):
        a = run.attesa_429(Errore429({"retry-after": "Wed, 21 Oct 2026 07:28:00 GMT"}), 1)
        self.assertTrue(2 <= a <= 3)


class InterruzioneTests(unittest.TestCase):
    def setUp(self):
        self.cartella = tempfile.TemporaryDirectory()
        base = Path(self.cartella.name)
        risposte().to_csv(base / "valutazione.csv", index=False, encoding="utf-8-sig")
        self.splits = mock.patch.object(run, "SPLITS", base)
        self.splits.start()
        self.percorso = base / "calls.jsonl"

    def tearDown(self):
        self.splits.stop()
        self.cartella.cleanup()

    def lancia(self, cfg, chiama):
        _, _, da_fare, totale = run.pianifica(cfg, common.leggi_calls(self.percorso))
        with mock.patch.object(run, "chiama", side_effect=chiama):
            run.esegui_lavori(da_fare, {}, cfg, common.tutte_le_etichette(cfg), contesto(cfg), self.percorso)
        return totale

    def righe(self):
        return [json.loads(r) for r in self.percorso.read_text(encoding="utf-8").splitlines() if r.strip()]

    def test_chiamata_interrotta_non_lascia_righe_e_viene_rifatta(self):
        for n_thread in (1, 4):
            with self.subTest(n_thread=n_thread):
                self.percorso.unlink(missing_ok=True)
                cfg = config(n_thread)
                interrotta = threading.Event()

                def chiama(m, clienti, messaggi, cfg_, testo, etichette):
                    if m["nome"] == "claude-finto" and testo == "risposta 3" and not interrotta.is_set():
                        interrotta.set()
                        raise KeyboardInterrupt  # come Ctrl+C a metà di questa chiamata
                    time.sleep(0.002)
                    return ok()

                with self.assertRaises(KeyboardInterrupt):
                    self.lancia(cfg, chiama)
                prima = self.righe()  # ogni riga è JSON completo
                self.assertLess(len(prima), 16)
                self.assertTrue(all(r["valido"] for r in prima))
                self.assertNotIn(("claude-finto", "3", 1), {(r["modello"], r["id"], r["run"]) for r in prima})

                totale = self.lancia(cfg, chiama)  # rilancio senza interruzioni
                dopo = self.righe()
                self.assertEqual(len(dopo), totale)
                self.assertEqual(len({chiave(r) for r in dopo}), totale)  # nessuna chiamata doppia
                self.assertIn(("claude-finto", "3", 1), {(r["modello"], r["id"], r["run"]) for r in dopo})

    def test_riga_troncata_viene_saltata_e_la_chiamata_rifatta(self):
        cfg = config(1)
        self.lancia(cfg, lambda *a, **k: ok())
        righe = self.percorso.read_text(encoding="utf-8").splitlines()
        troncata = json.loads(righe[-1])
        # processo ucciso mentre scriveva l'ultima riga: resta metà riga, senza a capo
        self.percorso.write_text("\n".join(righe[:-1]) + "\n" + righe[-1][:40], encoding="utf-8")

        with mock.patch("sys.stderr"):
            self.assertEqual(len(common.leggi_calls(self.percorso)), 15)
            totale = self.lancia(cfg, lambda *a, **k: ok())
            dopo = common.leggi_calls(self.percorso)
        self.assertEqual(len(dopo), totale)
        self.assertIn(chiave(troncata), {chiave(r) for r in dopo})


if __name__ == "__main__":
    unittest.main()
