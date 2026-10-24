import contextlib
import io
import runpy
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import common


class CitationVerificationTests(unittest.TestCase):
    def test_id_825_is_approximate_with_expected_score(self):
        result = common.verifica_citazione(
            "mi è piaciuto tutto",
            "min è piaciuto tutto soprattutto quando abbiamo fatto i percorsi",
            90,
        )
        self.assertEqual(result[0], "approssimata")
        self.assertAlmostEqual(result[1], 94.7, places=1)
        self.assertTrue(result[2].startswith("min è piaciuto tutt"))

    def test_literal_quote_is_exact(self):
        result = common.verifica_citazione("svolta dell'attività", "La svolta dell'attività è piaciuta", 90)
        self.assertEqual(result[0], "esatta")
        self.assertEqual(result[1], 100.0)

    def test_typographic_apostrophe_matches_ascii_apostrophe(self):
        result = common.verifica_citazione("l’attività preferita", "l'attività preferita", 90)
        self.assertEqual(result[0], "esatta")
        self.assertEqual(result[2], "l'attività preferita")

    def test_invented_quote_is_not_found(self):
        result = common.verifica_citazione("astronave arcobaleno segreta", "fare i percorsi tra i banchi", 90)
        self.assertEqual(result[0], "non_trovata")
        self.assertFalse(result[3])

    def test_empty_non_classifiable_quote_is_excluded(self):
        self.assertEqual(common.citazioni_non_vuote(""), [])
        self.assertEqual(common.citazioni_non_vuote(None), [])
        self.assertEqual(common.citazioni_non_vuote(["", "   ", "testo"]), ["testo"])

    def test_reservoir_records_do_not_enter_evaluation_metrics(self):
        cfg = common.carica_config()
        cfg["n_run"] = 3
        cfg["soglie_sensibilita"] = cfg["verifica_citazioni"]["soglie_sensibilita"]
        evaluation_text = "min è piaciuto tutto"
        reservoir_text = "lavorare con i compagni"
        records = []
        for run, quote in enumerate(
            ["mi è piaciuto tutto", "min è piaciuto tutto", ""], start=1
        ):
            records.append({
                "ts": f"2026-10-21T00:00:0{run}+00:00",
                "esperimento_id": "test_citazioni",
                "set": "valutazione",
                "provider": "mock",
                "modello": "modello_test",
                "strategia": "zero_shot",
                "run": run,
                "id": "825",
                "n_run": 3,
                "valido": True,
                "categoria": "Activity",
                "quote": quote,
                "motivazione": "test",
            })
            records.append({
                "ts": f"2026-10-21T00:01:0{run}+00:00",
                "esperimento_id": "test_citazioni",
                "set": "serbatoio",
                "provider": "mock",
                "modello": "modello_test",
                "strategia": "zero_shot",
                "run": run,
                "id": "r1",
                "n_run": 3,
                "valido": True,
                "categoria": "Social",
                "quote": "citazione inventata",
                "motivazione": "test",
            })

        original = {
            "carica_config": common.carica_config,
            "leggi_calls": common.leggi_calls,
            "SPLITS": common.SPLITS,
            "RESULTS": common.RESULTS,
            "METRICS": common.METRICS,
        }
        try:
            with tempfile.TemporaryDirectory() as directory:
                temp = Path(directory)
                splits = temp / "splits"
                results = temp / "results"
                metrics = results / "metrics"
                splits.mkdir()
                results.mkdir()
                pd.DataFrame([{"id": "825", "testo": evaluation_text, "umano": "Activity"}]).to_csv(
                    splits / "valutazione.csv", index=False, encoding="utf-8-sig"
                )
                pd.DataFrame([{"id": "r1", "testo": reservoir_text, "umano": "Social"}]).to_csv(
                    splits / "serbatoio.csv", index=False, encoding="utf-8-sig"
                )
                common.carica_config = lambda: cfg
                common.leggi_calls = lambda: records
                common.SPLITS = splits
                common.RESULTS = results
                common.METRICS = metrics
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(SCRIPTS / "04_aggregate.py"), run_name="__main__")
                    runpy.run_path(str(SCRIPTS / "05_metrics.py"), run_name="__main__")

                citations = pd.read_csv(results / "citazioni.csv", encoding="utf-8-sig")
                summary = pd.read_csv(metrics / "riepilogo.csv", encoding="utf-8-sig")
                aggregate = pd.read_csv(results / "aggregated.csv", encoding="utf-8-sig")
                evaluation_metrics = summary[summary["modello"] == "modello_test"].iloc[0]
                self.assertEqual(set(citations["set"]), {"valutazione", "serbatoio"})
                self.assertEqual(set(citations["versione_verifica"]), {"verifica_v2_tre_esiti"})
                self.assertTrue({"punteggio_somiglianza", "tratto_simile"} <= set(citations.columns))
                self.assertEqual(len(aggregate), 1)
                self.assertEqual(int(evaluation_metrics["n_citazioni"]), 2)
                self.assertEqual(int(evaluation_metrics["n_citazioni_approssimate"]), 1)
                self.assertEqual(int(evaluation_metrics["n_citazioni_non_trovate"]), 0)
                self.assertAlmostEqual(evaluation_metrics["tasso_citazioni_approssimate"], 0.5)
                sensitivity = pd.read_csv(metrics / "sensibilita_citazioni.csv", encoding="utf-8-sig")
                self.assertEqual(set(sensitivity["soglia"]), {80, 85, 90, 95})
                self.assertEqual(set(sensitivity["set"]), {"valutazione", "serbatoio"})
                self.assertTrue((sensitivity["modello"] == "modello_test").all())
        finally:
            for name, value in original.items():
                setattr(common, name, value)


if __name__ == "__main__":
    unittest.main()
