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
from metriche import accordo, etichetta_modale, gruppo_stabilita, id_con_etichetta_per_tutti

from common import ABBASTANZA_STABILE, INSTABILE, STABILE
ETICHETTE = ["Social", "Making", "non classificabile"]


def esempio():
    """4 risposte, 2 modelli. B non ha etichetta modale per l'id 2; A sbaglia l'id 4."""
    umano = {1: "Social", 2: "Social", 3: "Making", 4: "Making"}
    llm = {
        "A": {1: "Social", 2: "Social", 3: "Making", 4: "Social"},
        "B": {1: "Social", 2: None, 3: "Making", 4: "Making"},
    }
    return pd.DataFrame([
        {"esperimento_id": "e", "strategia": "zero_shot", "modello": m, "id": i,
         "umano": umano[i], "etichetta_modale": etichette[i]}
        for m, etichette in llm.items() for i in umano
    ])


class TreVersioniTests(unittest.TestCase):
    def setUp(self):
        self.agg = esempio()
        self.comuni = id_con_etichetta_per_tutti(self.agg)[("e", "zero_shot")]
        self.a = self.agg[self.agg["modello"] == "A"]
        self.b = self.agg[self.agg["modello"] == "B"]

    def test_insieme_comune(self):
        self.assertEqual(self.comuni, {1, 3, 4})

    def test_principale_esclude_e_riporta_copertura(self):
        r = accordo(self.b, ETICHETTE, "principale", self.comuni)
        self.assertEqual((r["n"], r["n_totale"]), (3, 4))
        self.assertAlmostEqual(r["copertura"], 0.75)
        self.assertAlmostEqual(r["kappa"], 1.0)
        self.assertAlmostEqual(r["f1_macro"], 1.0)

    def test_controllo_conta_errore_solo_nel_richiamo(self):
        r = accordo(self.b, ETICHETTE, "controllo", self.comuni)
        self.assertEqual(r["n"], 4)
        self.assertAlmostEqual(r["accordo_percentuale"], 0.75)
        # po = 0.75, pe = 0.5·0.25 + 0.5·0.5 = 0.375 → κ = 0.375 / 0.625
        self.assertAlmostEqual(r["kappa"], 0.6)
        self.assertAlmostEqual(r["po"], 0.75)
        self.assertAlmostEqual(r["pe"], 0.375)
        self.assertAlmostEqual((r["po"] - r["pe"]) / (1 - r["pe"]), r["kappa"])
        social, making = ETICHETTE.index("Social"), ETICHETTE.index("Making")
        self.assertAlmostEqual(r["recall"][social], 0.5)
        self.assertAlmostEqual(r["precision"][social], 1.0)
        self.assertAlmostEqual(r["precision"][making], 1.0)  # non assegnata a un'altra categoria
        self.assertAlmostEqual(r["f1_macro"], (2 / 3 + 1) / 2)

    def test_modello_con_tutte_le_etichette_ha_principale_uguale_a_controllo(self):
        p = accordo(self.a, ETICHETTE, "principale", self.comuni)
        c = accordo(self.a, ETICHETTE, "controllo", self.comuni)
        self.assertEqual((p["n"], c["n"]), (4, 4))
        self.assertAlmostEqual(p["accordo_percentuale"], c["accordo_percentuale"])
        self.assertAlmostEqual(p["kappa"], c["kappa"])

    def test_confronto_modelli_usa_lo_stesso_insieme(self):
        a = accordo(self.a, ETICHETTE, "confronto_modelli", self.comuni)
        b = accordo(self.b, ETICHETTE, "confronto_modelli", self.comuni)
        self.assertEqual(set(self.a.loc[a["y_umano"].index, "id"]), {1, 3, 4})
        self.assertEqual(set(self.b.loc[b["y_umano"].index, "id"]), {1, 3, 4})
        self.assertAlmostEqual(a["accordo_percentuale"], 2 / 3)
        self.assertAlmostEqual(b["accordo_percentuale"], 1.0)


class GruppiStabilitaTests(unittest.TestCase):
    def gruppo(self, categorie, n_run_attese=10):
        moda = etichetta_modale(categorie, n_run_attese)
        return moda, gruppo_stabilita(moda["etichetta_modale"], moda["quota_modale"])

    def test_10_su_10(self):
        moda, gruppo = self.gruppo(["Social"] * 10)
        self.assertEqual(moda["etichetta_modale"], "Social")
        self.assertEqual(gruppo, STABILE)

    def test_8_su_10(self):
        moda, gruppo = self.gruppo(["Social"] * 8 + ["Making"] * 2)
        self.assertEqual(moda["etichetta_modale"], "Social")
        self.assertEqual(gruppo, ABBASTANZA_STABILE)

    def test_7_su_10(self):
        moda, gruppo = self.gruppo(["Social"] * 7 + ["Making"] * 3)
        self.assertEqual(moda["etichetta_modale"], "Social")
        self.assertEqual(gruppo, INSTABILE)

    def test_pareggio_5_5_senza_etichetta(self):
        moda, gruppo = self.gruppo(["Social"] * 5 + ["Making"] * 5)
        self.assertIsNone(moda["etichetta_modale"])
        self.assertTrue(moda["pareggio"])
        self.assertEqual(gruppo, INSTABILE)

    def test_run_mancante_senza_etichetta_anche_se_unanime(self):
        moda, gruppo = self.gruppo(["Social"] * 9)
        self.assertIsNone(moda["etichetta_modale"])
        self.assertAlmostEqual(moda["quota_modale"], 0.9)  # rispetto alle run previste
        self.assertEqual(gruppo, INSTABILE)


class SerbatoioFuoriDalleMetricheTests(unittest.TestCase):
    def test_serbatoio_non_entra_in_nessuna_metrica(self):
        cfg = common.carica_config()
        cfg["n_run"] = 10
        valutazione = {"825": ("lavorare con i compagni", "Social"),
                       "826": ("costruire il robot", "Making")}
        records = []
        for set_nome, categoria_per_id in (
            ("valutazione", {"825": "Social", "826": "Making"}),
            # stesso id 825 e categorie sbagliate: se entrasse cambierebbe ogni metrica
            ("serbatoio", {"825": "Making", "s1": "Making"}),
        ):
            for id_, categoria in categoria_per_id.items():
                for run in range(1, 11):
                    records.append({
                        "ts": "2026-10-22T00:00:00+00:00", "esperimento_id": "test_set",
                        "set": set_nome, "provider": "mock", "modello": "modello_test",
                        "strategia": "zero_shot", "run": run, "id": id_, "n_run": 10,
                        "valido": True, "categoria": categoria, "quote": "", "motivazione": "test",
                    })

        original = {name: getattr(common, name)
                    for name in ("carica_config", "leggi_calls", "SPLITS", "RESULTS", "METRICS")}
        try:
            with tempfile.TemporaryDirectory() as directory:
                temp = Path(directory)
                splits, results = temp / "splits", temp / "results"
                metrics = results / "metrics"
                splits.mkdir()
                results.mkdir()
                pd.DataFrame([{"id": i, "testo": t, "umano": u} for i, (t, u) in valutazione.items()]).to_csv(
                    splits / "valutazione.csv", index=False, encoding="utf-8-sig")
                pd.DataFrame([{"id": "825", "testo": "x", "umano": "Making"},
                              {"id": "s1", "testo": "y", "umano": "Making"}]).to_csv(
                    splits / "serbatoio.csv", index=False, encoding="utf-8-sig")
                common.carica_config = lambda: cfg
                common.leggi_calls = lambda: records
                common.SPLITS, common.RESULTS, common.METRICS = splits, results, metrics
                with contextlib.redirect_stdout(io.StringIO()):
                    runpy.run_path(str(SCRIPTS / "04_aggregate.py"), run_name="__main__")
                    runpy.run_path(str(SCRIPTS / "05_metrics.py"), run_name="__main__")

                aggregate = pd.read_csv(results / "aggregated.csv", dtype={"id": str}, encoding="utf-8-sig")
                summary = pd.read_csv(metrics / "riepilogo.csv", encoding="utf-8-sig").iloc[0]
                stability = pd.read_csv(metrics / "stabilita.csv", encoding="utf-8-sig")
                columns = set(pd.read_csv(metrics / "riepilogo.csv", encoding="utf-8-sig").columns)
                thesis = pd.read_csv(metrics / "tabella_tesi.csv", encoding="utf-8-sig")
                self.assertFalse({"n_controllo", "copertura_controllo", "n_risposte_run_escluse",
                                  "quota_risposte_con_tutte_le_citazioni_verificate"} & columns)
                self.assertFalse(any("accuratezza" in c for c in columns | set(stability.columns)))
                self.assertEqual([c for c in columns if c.startswith("n_categorie_f1_macro")],
                                 ["n_categorie_f1_macro"])
                for versione in ("principale", "controllo", "confronto_modelli"):
                    self.assertTrue({f"po_{versione}", f"pe_{versione}"} <= columns)
                self.assertEqual(list(thesis.columns), [
                    "modello", "strategia", "copertura_principale", "kappa_cohen_principale",
                    "accordo_percentuale_principale", "f1_macro_principale", "kappa_cohen_controllo",
                    "kappa_cohen_confronto_modelli", "ta_at_n", "kappa_fleiss",
                    "tasso_citazioni_approssimate", "tasso_citazioni_non_trovate",
                ])
                self.assertEqual(len(thesis), 1)
                per_category = pd.read_csv(metrics / "per_categoria.csv", encoding="utf-8-sig")

                self.assertEqual(sorted(aggregate["id"]), ["825", "826"])
                self.assertTrue((aggregate["quota_modale"] == 1.0).all())
                self.assertEqual(int(summary["n_risposte"]), 2)
                self.assertEqual(int(summary["n_risposte_run_complete"]), 2)
                self.assertAlmostEqual(summary["ta_at_n"], 1.0)
                for versione in ("principale", "confronto_modelli"):
                    self.assertEqual(int(summary[f"n_{versione}"]), 2)
                for versione in ("principale", "controllo", "confronto_modelli"):
                    self.assertAlmostEqual(summary[f"accordo_percentuale_{versione}"], 1.0)
                    self.assertAlmostEqual(summary[f"kappa_cohen_{versione}"], 1.0)
                self.assertEqual(int(stability.loc[stability["gruppo_stabilita"] == STABILE, "n_gruppo"].iloc[0]), 2)
                self.assertEqual(int(per_category.groupby("versione")["supporto"].sum().max()), 2)
        finally:
            for name, value in original.items():
                setattr(common, name, value)


if __name__ == "__main__":
    unittest.main()
