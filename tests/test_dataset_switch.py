# -*- coding: utf-8 -*-
"""Both panels load, and dataset 1 still loads exactly what it always did.

The point of these tests is the second half. Adding a second panel is only safe if
the first one is untouched, so the dataset-1 numbers are asserted against the
figures the repository already documents (16,986 issuer-months, 293 issuers, 32
positive months from 8 issuers). If a future change to the adapter alters them,
this fails rather than quietly re-basing the research.

The dataset-2 numbers are asserted against the independent run recorded in the
leadning_time project (187,007 firm-months, 941 firms, y_pre3m = 1 in 124 rows
from 31 firms). Two code bases agreeing on those counts is what makes the load
trustworthy. Those tests skip when lime_credit.db is not on the machine, since it
is 127 MB and cannot be committed.
"""
import unittest

import ibond_dataset as ds
import cmdf_tree_classify as cl


def _have(choice):
    ds.use(choice)
    try:
        ds.require_db()
        return True
    except FileNotFoundError:
        return False


class DatasetSelection(unittest.TestCase):
    def tearDown(self):
        ds.use(1)

    def test_default_is_dataset_one(self):
        ds.use(1)
        self.assertEqual(ds.TABLE, "ibond_33features_panel")

    def test_table_names(self):
        self.assertEqual(ds.DATASETS[1]["table"], "ibond_33features_panel")
        self.assertEqual(ds.DATASETS[2]["table"],
                         "ibond_33features_panel_941firm")

    def test_result_tables_do_not_collide(self):
        ds.use(1)
        one = ds.tname("cmdf_pairwise_shock")
        ds.use(2)
        two = ds.tname("cmdf_pairwise_shock")
        self.assertEqual(one, "cmdf_pairwise_shock")
        self.assertEqual(two, "cmdf_pairwise_shock_ds2")
        self.assertNotEqual(one, two)

    def test_output_folders_do_not_collide(self):
        ds.use(1)
        one = ds.OUTDIR
        ds.use(2)
        self.assertNotEqual(one, ds.OUTDIR)

    def test_results_are_written_to_one_database(self):
        ds.use(1)
        one = ds.RESULT_DB
        ds.use(2)
        self.assertEqual(one, ds.RESULT_DB)
        self.assertTrue(ds.RESULT_DB.endswith("cmdf_credit.db"))


class PanelContents(unittest.TestCase):
    def tearDown(self):
        ds.use(1)

    @unittest.skipUnless(_have(1), "cmdf_credit.db not present")
    def test_dataset_one_is_unchanged(self):
        panel, X, y, cols = cl.load_panel(verbose=False, dataset=1)
        self.assertEqual(len(panel), 16986)
        self.assertEqual(panel["issuer_code"].nunique(), 293)
        self.assertEqual(len(cols), 33)
        self.assertEqual(int(y.sum()), 32)
        self.assertEqual(panel.loc[y == 1, "issuer_code"].nunique(), 8)

    @unittest.skipUnless(_have(2), "lime_credit.db not present")
    def test_dataset_two_matches_the_independent_run(self):
        panel, X, y, cols = cl.load_panel(verbose=False, dataset=2)
        self.assertEqual(len(panel), 187007)
        self.assertEqual(panel["issuer_code"].nunique(), 941)
        self.assertEqual(len(cols), 33)
        self.assertEqual(int(y.sum()), 124)
        self.assertEqual(panel.loc[y == 1, "issuer_code"].nunique(), 31)

    @unittest.skipUnless(_have(2), "lime_credit.db not present")
    def test_both_panels_expose_the_same_determinants(self):
        _, _, _, c1 = cl.load_panel(verbose=False, dataset=1)
        _, _, _, c2 = cl.load_panel(verbose=False, dataset=2)
        self.assertEqual(c1, c2)


if __name__ == "__main__":
    unittest.main()
