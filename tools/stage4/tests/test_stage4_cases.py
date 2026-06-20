from __future__ import annotations

import unittest

from tools.stage4.stage4_ogm_cases import build_fixed_eval_cases, split_case_counts


class Stage4FixedCaseTests(unittest.TestCase):
    def test_fixed_eval_cases_have_20_parallel_and_50_perpendicular(self) -> None:
        cases = build_fixed_eval_cases()
        counts = split_case_counts(cases)
        self.assertEqual(counts["parallel"], 20)
        self.assertEqual(counts["perpendicular"], 50)
        self.assertEqual(len(cases), 70)

    def test_fixed_eval_cases_have_stable_ids_and_hope_case_ids(self) -> None:
        cases = build_fixed_eval_cases()
        self.assertEqual(cases[0]["case_uid"], "parallel_000")
        self.assertEqual(cases[0]["hope_case_id"], 1)
        self.assertEqual(cases[20]["case_uid"], "perpendicular_000")
        self.assertEqual(cases[20]["hope_case_id"], 0)
        self.assertEqual(len({case["seed"] for case in cases}), 70)


if __name__ == "__main__":
    unittest.main()
