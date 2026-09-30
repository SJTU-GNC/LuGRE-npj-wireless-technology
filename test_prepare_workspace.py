"""Standard-library safety and transformation tests; no scientific runs."""
import ast
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import prepare_workspace as prep


class PreparationTests(unittest.TestCase):
    def test_reject_original_and_nonempty(self):
        with self.assertRaises(ValueError):
            prep.validate_target(Path(prep.OLD_ROOT), [])
        with tempfile.TemporaryDirectory(prefix="lugre_prepare_test_") as tmp:
            root = Path(tmp)
            (root / "keep.txt").write_text("do not overwrite", encoding="utf-8")
            with self.assertRaises(ValueError):
                prep.validate_target(root, [])
            self.assertEqual((root / "keep.txt").read_text(), "do not overwrite")
            with self.assertRaises(ValueError):
                prep.validate_target(root / "child", [root])

    def test_known_non_numerical_transformations(self):
        text = ('from pathlib import Path\nROOT = Path(r"D:\\月球导航")\n'
                'inputs = [MODEL, OUT / "PLAN.md", Path(__file__)]\n'
                'provenance = [OUT / "PLAN.md", Path(__file__)]\n'
                'x = [RESPONSE, OUT.parent / "PLAN.md"]\n')
        result, changes = prep.transform_python(text, "test.py", Path("E:/portable"))
        self.assertNotIn("PLAN.md", result)
        self.assertIn("E:/portable", result)
        self.assertEqual(next(item["count"] for item in changes if item["kind"] == "remove_internal_plan_hash_dependency"), 3)
        source = '    rows = pd.read_csv(OUT / "attrib_rows_empirical_fit32.csv", float_precision="round_trip")'
        result, _ = prep.transform_python("def main():\n" + source + "\n",
            "analysis/phase_compensation_attribution_v4/attrib_phase_interpretation.py", Path("E:/portable"))
        self.assertIn('rows["reference"].eq("empirical_fit32")', result)

    def test_copy_only_new_workspace_and_record_hashes(self):
        with tempfile.TemporaryDirectory(prefix="lugre_prepare_test_") as tmp:
            root = Path(tmp)
            sources, data, target = root / "sources", root / "data", root / "workspace"
            (sources / "script").mkdir(parents=True)
            data.mkdir()
            code = b'from pathlib import Path\nROOT = Path("D:/\xe6\x9c\x88\xe7\x90\x83\xe5\xaf\xbc\xe8\x88\xaa")\nVALUE = 0.006514909\n'
            (sources / "script/example.py").write_bytes(code)
            numeric = b"x,y\n1,2.5\n"
            (data / "sample.csv").write_bytes(numeric)
            argv = ["prepare_workspace.py", "--sources", str(sources), "--data", str(data), "--workspace", str(target)]
            with patch("sys.argv", argv), contextlib.redirect_stdout(io.StringIO()):
                prep.main()
            self.assertEqual((sources / "script/example.py").read_bytes(), code)
            self.assertEqual((target / "sample.csv").read_bytes(), numeric)
            log = json.loads((target / prep.LOG_NAME).read_text(encoding="utf-8"))
            self.assertEqual(log["file_count"], 2)
            self.assertEqual(log["changed_file_count"], 1)
            self.assertFalse(log["original_inputs_modified"])
            with patch("sys.argv", argv), self.assertRaises(ValueError):
                prep.main()

    def test_all_released_sources_preserve_scientific_numbers(self):
        sources = Path(__file__).resolve().parent / "original_sources"
        if not sources.is_dir():
            self.skipTest("original_sources not assembled")

        def numbers(text):
            lines = text.splitlines()
            return [node.value for node in ast.walk(ast.parse(text))
                    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float, complex))
                    and not isinstance(node.value, bool)
                    and "parents[" not in lines[node.lineno - 1]]

        for source in sources.rglob("*.py"):
            relative = source.relative_to(sources).as_posix()
            original = source.read_text(encoding="utf-8-sig")
            transformed, _ = prep.transform_python(original, relative, Path("E:/portable"))
            with self.subTest(path=relative):
                self.assertEqual(numbers(original), numbers(transformed))


if __name__ == "__main__":
    unittest.main()
