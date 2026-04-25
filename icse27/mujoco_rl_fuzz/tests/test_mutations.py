"""Lightweight tests; runnable with: python -m unittest discover tests"""
import os
import random
import tempfile
import unittest

from lxml import etree

from src.mutations.registry import MUTATORS, MUTATOR_IDS

SEED_XML = os.path.join(os.path.dirname(__file__), "..", "seeds", "pendulum.xml")


class TestMutatorsApplyAndCompile(unittest.TestCase):
    def test_each_mutator_apply_writes_valid_xml(self):
        rng = random.Random(0)
        for mid in MUTATOR_IDS:
            with self.subTest(mid=mid):
                m = MUTATORS[mid]
                tree = etree.parse(SEED_XML)
                if not m.applicable(tree):
                    continue
                params = m.sample_params(tree, rng)
                with tempfile.TemporaryDirectory() as td:
                    out = os.path.join(td, f"{mid}.xml")
                    res = m.apply(tree, params, out)
                    if not res.ok:
                        self.assertIsNotNone(res.reason)
                        continue
                    self.assertTrue(os.path.exists(res.new_xml_path),
                                    f"{mid}: new_xml_path missing: {res.new_xml_path}")
                    etree.parse(res.new_xml_path)  # well-formed


if __name__ == "__main__":
    unittest.main()
