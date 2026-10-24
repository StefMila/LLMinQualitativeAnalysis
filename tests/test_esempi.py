import importlib
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common

esempi_mod = importlib.import_module("01b_esempi")
prompt_mod = importlib.import_module("02_prompt")
ETICHETTE = ["Social", "Making", "Activity", "non classificabile"]


def split(righe):
    return pd.DataFrame(righe, columns=["id", "testo", "umano"]).astype({"id": str})


class SceltaEsempiTests(unittest.TestCase):
    def setUp(self):
        self.serbatoio = split([
            ("1", "Lavorare  in GRUPPO", "Social"),   # uguale a una di valutazione dopo normalizzazione
            ("2", "gli amici", "Social"),
            ("3", "costruire", "Making"),
            ("4", "provare", "Making"),
            ("5", "sperimentare", "Making"),
            ("6", "fare cose", "Making"),
            ("7", "il robot", "Activity"),
        ])
        self.valutazione = split([("100", "lavorare in gruppo", "Social"), ("101", "boh", "non classificabile")])
        self.ordine, self.esclusi = esempi_mod.ordine_esempi(self.serbatoio, self.valutazione, ETICHETTE, 42)

    def test_esclude_i_testi_uguali_alla_valutazione(self):
        self.assertEqual(list(self.esclusi["id"]), ["1"])
        self.assertNotIn("1", set(self.ordine["id"]))

    def test_categoria_assente_non_inclusa(self):
        self.assertNotIn("non classificabile", set(self.ordine["categoria"]))

    def test_ordine_riproducibile_e_nell_ordine_del_codebook(self):
        di_nuovo, _ = esempi_mod.ordine_esempi(self.serbatoio, self.valutazione, ETICHETTE, 42)
        pd.testing.assert_frame_equal(self.ordine, di_nuovo)
        self.assertEqual(list(dict.fromkeys(self.ordine["categoria"])), ["Social", "Making", "Activity"])

    def test_estrazione_annidata_e_massimo_tre(self):
        one = esempi_mod.esempi_strategia(self.ordine, "one_shot")
        few = esempi_mod.esempi_strategia(self.ordine, "few_shot")
        self.assertEqual(few.groupby("categoria").size().to_dict(), {"Social": 1, "Making": 3, "Activity": 1})
        primi = self.ordine[self.ordine["rango"] == 1]
        self.assertEqual(list(one["id"]), list(primi["id"]))
        self.assertTrue(set(one["id"]) <= set(few["id"]))
        self.assertEqual(list(few["posizione"]), list(range(1, len(few) + 1)))


class EsempiSalvatiTests(unittest.TestCase):
    """Controlla i file in prompts/esempi/<domanda>/ prodotti da 01b_esempi.py."""

    def setUp(self):
        self.cfg = common.carica_config()
        self.valutazione = pd.read_csv(common.SPLITS / "valutazione.csv", dtype={"id": str}, encoding="utf-8-sig")
        self.serbatoio = pd.read_csv(common.SPLITS / "serbatoio.csv", dtype={"id": str}, encoding="utf-8-sig")
        self.cartella = common.PROMPTS / "esempi" / common.DOMANDA

    def leggi(self, nome):
        return pd.read_csv(self.cartella / nome, dtype={"id": str}, encoding="utf-8-sig")

    def test_nessun_esempio_nel_campione_di_valutazione(self):
        for nome in ("ordine_esempi.csv", "esempi_one_shot.csv", "esempi_few_shot.csv"):
            self.assertFalse(set(self.leggi(nome)["id"]) & set(self.valutazione["id"]), nome)
            self.assertTrue(set(self.leggi(nome)["id"]) <= set(self.serbatoio["id"]), nome)

    def test_file_coincidono_con_la_regola_e_il_seed(self):
        ordine, _ = esempi_mod.ordine_esempi(
            self.serbatoio, self.valutazione, common.tutte_le_etichette(self.cfg), self.cfg["esempi"]["seed"]
        )
        salvato = self.leggi("ordine_esempi.csv")
        self.assertEqual(list(salvato["id"]), list(ordine["id"]))
        for strategia in ("one_shot", "few_shot"):
            atteso = esempi_mod.esempi_strategia(ordine, strategia)
            self.assertEqual(list(self.leggi(f"esempi_{strategia}.csv")["id"]), list(atteso["id"]))


class PromptFewShotTests(unittest.TestCase):
    def setUp(self):
        self.cfg = common.carica_config()

    def test_sistema_identico_allo_zero_shot_ed_esempi_fissi(self):
        zero = prompt_mod.costruisci_messaggi("prima risposta", "zero_shot", self.cfg)
        few_1 = prompt_mod.costruisci_messaggi("prima risposta", "few_shot", self.cfg)
        few_2 = prompt_mod.costruisci_messaggi("un'altra risposta", "few_shot", self.cfg)
        self.assertEqual(few_1[0], zero[0])
        self.assertEqual(len(few_1), 2)
        esempi_1, utente_1 = few_1[1]["content"]
        esempi_2, utente_2 = few_2[1]["content"]
        self.assertEqual(esempi_1, esempi_2)  # prefisso identico in ogni chiamata
        self.assertEqual(utente_1["text"], zero[1]["content"])  # risposta da codificare come nello zero-shot
        self.assertIn("un'altra risposta", utente_2["text"])

    def test_esempi_nel_prompt_in_ordine_senza_motivazioni(self):
        esempi = pd.read_csv(
            common.PROMPTS / "esempi" / common.DOMANDA / "esempi_few_shot.csv", encoding="utf-8-sig"
        )
        testo = prompt_mod.blocco_esempi("few_shot")
        posizioni = [testo.index(f'- "{" ".join(t.split())}"') for t in esempi.sort_values("posizione")["testo"]]
        self.assertEqual(posizioni, sorted(posizioni))
        intestazioni = [r[3:] for r in testo.splitlines() if r.startswith("## ")]
        ordine_codebook = [e for e in common.tutte_le_etichette(self.cfg) if e in intestazioni]
        self.assertEqual(intestazioni, ordine_codebook)
        self.assertNotIn("motivazione", testo.lower())
        self.assertNotIn("quote", testo.lower())


if __name__ == "__main__":
    unittest.main()
