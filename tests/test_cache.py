"""La cache cambia solo la configurazione della chiamata a Claude, mai il testo inviato."""
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common

run = importlib.import_module("03_run")
prompt_mod = importlib.import_module("02_prompt")


class FakeAnthropicClient:
    def __init__(self):
        self.request = None
        self.messages = self

    def create(self, **kwargs):
        self.request = kwargs
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="{}")],
            usage=SimpleNamespace(
                input_tokens=10, output_tokens=200, cache_creation_input_tokens=0,
                cache_read_input_tokens=2000, output_tokens_details=SimpleNamespace(thinking_tokens=120),
            ),
            model="claude-opus-5-test",
        )


def testo_inviato(request):
    """Ricostruisce il prompt nel formato di prompt_esempio_<strategia>.txt, ignorando cache_control."""
    system = request["system"]
    if not isinstance(system, str):
        system = "\n\n".join(blocco["text"] for blocco in system)
    parti = [("system", system)] + [
        (m["role"], prompt_mod.testo_messaggio(m["content"])) for m in request["messages"]
    ]
    return "\n".join(f"===== {ruolo} =====\n{testo}\n" for ruolo, testo in parti)


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cfg = common.carica_config()
        self.riga = pd.read_csv(common.SPLITS / "valutazione.csv", encoding="utf-8-sig").iloc[0]

    def invia(self, strategia, cache):
        client = FakeAnthropicClient()
        messaggi = prompt_mod.costruisci_messaggi(self.riga["testo"], strategia, self.cfg)
        _, _, token = run.chiama_anthropic(client, "claude-opus-5", messaggi, None, "high", cache=cache)
        return client.request, token

    def test_testo_identico_al_prompt_congelato(self):
        congelato = (common.PROMPTS / "esempi" / common.DOMANDA / "prompt_esempio_few_shot.txt").read_text(
            encoding="utf-8"
        )
        senza, _ = self.invia("few_shot", cache=False)
        con, _ = self.invia("few_shot", cache=True)
        self.assertEqual(testo_inviato(senza), testo_inviato(con))
        self.assertEqual(testo_inviato(con), congelato)

    def test_segnaposto_solo_dopo_gli_esempi(self):
        con, _ = self.invia("few_shot", cache=True)
        blocchi = con["messages"][0]["content"]
        self.assertEqual(blocchi[0]["cache_control"], {"type": "ephemeral"})
        self.assertNotIn("cache_control", blocchi[1])  # la risposta dello studente resta fuori
        self.assertTrue(blocchi[1]["text"].startswith("Domanda:"))
        self.assertIsInstance(con["system"], str)

    def test_senza_cache_nessun_segnaposto(self):
        senza, _ = self.invia("few_shot", cache=False)
        self.assertNotIn("cache_control", str(senza))

    def test_zero_shot_cache_sul_system(self):
        con, _ = self.invia("zero_shot", cache=True)
        self.assertEqual(con["system"][0]["cache_control"], {"type": "ephemeral"})
        self.assertIsInstance(con["messages"][0]["content"], str)

    def test_token_registrati(self):
        _, token = self.invia("few_shot", cache=True)
        self.assertEqual(token, {
            "token_in": 2010, "token_out": 200, "token_cache_scrittura": 0,
            "token_cache_lettura": 2000, "token_ragionamento": 120,
        })

    def test_messaggi_originali_non_modificati(self):
        messaggi = prompt_mod.costruisci_messaggi(self.riga["testo"], "few_shot", self.cfg)
        run.chiama_anthropic(FakeAnthropicClient(), "claude-opus-5", messaggi, None, "high", cache=True)
        self.assertNotIn("cache_control", str(messaggi))


if __name__ == "__main__":
    unittest.main()
