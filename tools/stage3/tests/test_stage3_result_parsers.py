import tempfile
import unittest
from pathlib import Path

from tools.stage3.stage3_result_parsers import (
    collect_eval_success,
    parse_eval_result_file,
    summarize_resource_csv,
)


class Stage3ResultParserTests(unittest.TestCase):
    def test_parse_eval_result_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.txt"
            result_path.write_text("success rate: 0.945\nstep num: 88.1 +-(12.3)\n", encoding="utf-8")
            parsed = parse_eval_result_file(result_path)
            self.assertEqual(parsed["success_rate"], 0.945)
            self.assertEqual(parsed["step_num_mean"], 88.1)

    def test_parse_eval_result_file_requires_success_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.txt"
            result_path.write_text("step num: 88.1 +-(12.3)\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                parse_eval_result_file(result_path)

    def test_collect_eval_success_from_scene_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scene_dir, value in {"normalize": 0.985, "complex": 0.945, "extreme": 0.655, "dlp": 0.960}.items():
                path = root / scene_dir
                path.mkdir()
                (path / "result.txt").write_text(f"success rate: {value}\nstep num: 90.0 +-(1.0)\n", encoding="utf-8")
            success = collect_eval_success(root)
            self.assertEqual(success, {"Normal": 0.985, "Complex": 0.945, "Extrem": 0.655, "DLP": 0.960})

    def test_summarize_resource_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.csv"
            path.write_text(
                "timestamp,pid,process_name,cpu_percent,working_set_mb,private_memory_mb,gpu_util_percent,gpu_memory_used_mb,gpu_memory_total_mb,gpu_power_w\n"
                "2026-06-19T00:00:00,1,python,10,100,80,20,1000,12000,50\n"
                "2026-06-19T00:00:05,1,python,30,120,90,40,1500,12000,60\n",
                encoding="utf-8",
            )
            summary = summarize_resource_csv(path)
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["avg_process_cpu_percent"], 20.0)
            self.assertEqual(summary["max_process_cpu_percent"], 30.0)
            self.assertEqual(summary["avg_working_set_mb"], 110.0)
            self.assertEqual(summary["max_working_set_mb"], 120.0)
            self.assertEqual(summary["avg_private_memory_mb"], 85.0)
            self.assertEqual(summary["max_private_memory_mb"], 90.0)
            self.assertEqual(summary["avg_whole_gpu_util_percent"], 30.0)
            self.assertEqual(summary["max_whole_gpu_util_percent"], 40.0)
            self.assertEqual(summary["avg_gpu_memory_used_mb"], 1250.0)
            self.assertEqual(summary["max_gpu_memory_used_mb"], 1500.0)
            self.assertEqual(summary["avg_gpu_memory_total_mb"], 12000.0)
            self.assertEqual(summary["max_gpu_memory_total_mb"], 12000.0)
            self.assertEqual(summary["avg_gpu_power_w"], 55.0)
            self.assertEqual(summary["max_gpu_power_w"], 60.0)


if __name__ == "__main__":
    unittest.main()
