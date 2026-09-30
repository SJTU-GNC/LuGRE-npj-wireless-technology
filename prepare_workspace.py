"""Prepare an isolated, portable working copy of the released code and data.

Run from release/code:
    python prepare_workspace.py --data ../data --workspace ../run_workspace

The original_sources directory and data archive are read-only inputs. This
program never trains a model, downloads raw public inputs, or changes numbers.
Only the new workspace receives the documented portability transformations.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import sys
import tokenize


OLD_ROOT = "D:/月球导航"
OLD_PROJECTS = (
    "fabio_xz/AI-driven cislunar GNSS channel modelling from lunar GNSS observations",
    "fabio_xz/AI-driven cislunar GNSS channel modelling - problem-driven revision 20260924",
)
LOG_NAME = "PORTABILITY_CHANGES.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_link(path: Path) -> bool:
    return path.is_symlink() or getattr(path, "is_junction", lambda: False)()


def validate_target(target: Path, inputs: list[Path]) -> None:
    """No overwrite, source nesting, drive-root, or original workspace writes."""
    if is_link(target):
        raise ValueError("The workspace must not be a symlink or junction.")
    destination = target.resolve()
    forbidden = [Path(OLD_ROOT).resolve()]
    forbidden += [(Path(OLD_ROOT) / value).resolve() for value in OLD_PROJECTS]
    if destination == Path(destination.anchor) or destination == Path.home().resolve():
        raise ValueError("Refusing a drive root or home directory as workspace.")
    if destination in forbidden or any(root.is_relative_to(destination) for root in forbidden):
        raise ValueError("Refusing the original workspace/project or an ancestor.")
    for source in inputs:
        source = source.resolve()
        if destination.is_relative_to(source) or source.is_relative_to(destination):
            raise ValueError("Workspace and input directories must not overlap.")
    if destination.exists():
        if not destination.is_dir() or any(destination.iterdir()):
            raise ValueError("Workspace already exists and is not an empty directory.")


def replace_once(text: str, old: str, new: str, changes: list[dict], reason: str) -> str:
    count = text.count(old)
    if count:
        if count != 1:
            raise ValueError(f"Expected one occurrence for {reason}, found {count}")
        changes.append({"kind": reason, "old": old, "new": new, "count": count})
        return text.replace(old, new)
    return text


def rebase_path_literals(text: str, workspace: Path, changes: list[dict]) -> str:
    """Change only complete Python string literals beginning with the old root."""
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.STRING:
            try:
                value = ast.literal_eval(token.string)
            except (ValueError, SyntaxError):
                value = None
            if isinstance(value, str):
                normalized = value.replace("\\", "/")
                prefixes = [OLD_ROOT + "/" + item for item in OLD_PROJECTS] + [OLD_ROOT]
                for prefix in prefixes:
                    if normalized == prefix or normalized.startswith(prefix + "/"):
                        new = workspace.as_posix() + normalized[len(prefix):]
                        changes.append({"kind": "absolute_path_rebase", "old": value, "new": new})
                        token = token._replace(string=repr(new))
                        break
        tokens.append(token)
    return tokenize.untokenize(tokens)


def transform_python(text: str, relative: str, workspace: Path) -> tuple[str, list[dict]]:
    changes: list[dict] = []
    for project in OLD_PROJECTS:
        for root in ("ROOT", "WORKSPACE"):
            for quote in ('"', "'"):
                text = replace_once(
                    text, f"MANUSCRIPT = {root} / {quote}{project}{quote}",
                    f"MANUSCRIPT = {root}", changes, "flatten_project_directory")
    if relative == "analysis/phase_residual_analysis_v1/prepare_physical_alignment.py":
        text = replace_once(text, "ROOT = Path(__file__).resolve().parents[4]",
                            "ROOT = Path(__file__).resolve().parents[2]", changes,
                            "flatten_workspace_depth")
    if relative == "analysis/thermal_timeseries_v15/build_diagnostics.py":
        text = replace_once(text, "WORKSPACE = PROJECT.parents[1]", "WORKSPACE = PROJECT",
                            changes, "flatten_workspace_depth")
    if relative == "analysis/phase_compensation_attribution_v4/attrib_phase_interpretation.py":
        old = '    rows = pd.read_csv(OUT / "attrib_rows_empirical_fit32.csv", float_precision="round_trip")'
        new = ('    rows = pd.read_csv(OUT / "attrib_rows.csv", float_precision="round_trip")\n'
               '    rows = rows.loc[rows["reference"].eq("empirical_fit32")].copy()')
        text = replace_once(text, old, new, changes, "use_deduplicated_reference_table")
    # These files hash an internal planning note, but never use its contents in
    # a numerical calculation. Keep hashes of all numerical inputs and code.
    plan_pattern = r'''(?:,\s*OUT(?:\.parent)?\s*/\s*(["'])PLAN\.md\1|OUT(?:\.parent)?\s*/\s*(["'])PLAN\.md\2\s*,\s*)'''
    text, count = re.subn(plan_pattern, "", text)
    if count:
        changes.append({"kind": "remove_internal_plan_hash_dependency", "count": count,
                        "note": "Numerical inputs and mathematical expressions unchanged."})
    text = rebase_path_literals(text, workspace, changes)
    compile(text, relative, "exec")
    if re.search(r'''OUT(?:\.parent)?\s*/\s*["']PLAN\.md''', text):
        raise ValueError(f"Unresolved internal PLAN.md dependency: {relative}")
    remaining_paths = text.replace("\\", "/").replace(workspace.as_posix(), "<WORKSPACE>")
    if OLD_ROOT in remaining_paths:
        raise ValueError(f"Unresolved original absolute root: {relative}")
    return text, changes


def transform_metadata(text: str, relative: str) -> tuple[str, list[dict]]:
    """Validate supplied provenance metadata without changing its contents."""
    changes: list[dict] = []
    if relative.endswith("alex_temperature_inputs/digitization_provenance.json"):
        json.loads(text)
    return text, changes


def collect(source: Path, label: str) -> list[tuple[Path, str, str]]:
    if not source.is_dir() or is_link(source):
        raise ValueError(f"Missing directory or disallowed link: {source}")
    files = []
    for path in sorted(source.rglob("*")):
        if is_link(path):
            raise ValueError(f"Input links/junctions are not copied: {path}")
        if path.is_file():
            files.append((path, path.relative_to(source).as_posix(), label))
    if not files:
        raise ValueError(f"Input directory is empty: {source}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    parser.add_argument("--workspace", type=Path, required=True,
                        help="A new or empty destination; never an original workspace.")
    parser.add_argument("--sources", type=Path, default=Path(__file__).resolve().parent / "original_sources")
    parser.add_argument("--dry-run", action="store_true", help="Preflight every transformation without writing files.")
    args = parser.parse_args()
    validate_target(args.workspace, [args.sources, args.data])
    workspace = args.workspace.resolve()
    sources, data = args.sources.resolve(), args.data.resolve()
    entries = collect(sources, "original_sources") + collect(data, "data")
    seen: set[str] = set()
    manifest, planned = [], []
    for source, relative, label in entries:
        key = relative.casefold()
        if key in seen or key == LOG_NAME.casefold():
            raise ValueError(f"Duplicate/reserved destination: {relative}")
        seen.add(key)
        original = source.read_bytes()
        changes: list[dict] = []
        result = original
        if source.suffix == ".py":
            text, changes = transform_python(original.decode("utf-8-sig"), relative, workspace)
            if changes:
                result = text.encode("utf-8")
        elif relative.endswith("alex_temperature_inputs/digitization_provenance.json"):
            text, changes = transform_metadata(original.decode("utf-8-sig"), relative)
            if changes:
                result = text.encode("utf-8")
        manifest.append({"path": relative, "input_group": label,
                         "source_sha256": digest(original), "workspace_sha256": digest(result),
                         "changes": changes})
        planned.append((source, relative, result if changes else None))
    report = {"schema_version": "1.0", "workspace": str(workspace),
              "sources": str(sources), "data": str(data), "file_count": len(entries),
              "changed_file_count": sum(bool(item["changes"]) for item in manifest),
              "original_inputs_modified": False, "model_training_performed": False,
              "scope": "Portability and public provenance wording only; no numerical transformations.",
              "limitations": ["Raw public products are not bundled or downloaded.",
                              "Preparation does not execute or validate every analysis/figure pipeline.",
                              "Legacy alex_* filenames, columns and scenario IDs are retained for numerical compatibility."],
              "files": manifest}
    if args.dry_run:
        print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
        return
    # All parsing/collision checks finish before the first write. A concurrent
    # creator cannot make us silently reuse a directory with existing content.
    validate_target(workspace, [sources, data])
    workspace.mkdir(parents=True, exist_ok=True)
    for source, relative, result in planned:
        destination = workspace / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as stream:
            if result is None:
                with source.open("rb") as input_stream:
                    shutil.copyfileobj(input_stream, stream)
            else:
                stream.write(result)
    for item, (source, _, _) in zip(manifest, planned):
        if digest(source.read_bytes()) != item["source_sha256"]:
            raise RuntimeError(f"An input changed during preparation: {source}")
        if digest((workspace / item["path"]).read_bytes()) != item["workspace_sha256"]:
            raise RuntimeError(f"Workspace integrity check failed: {item['path']}")
    with (workspace / LOG_NAME).open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(f"Prepared {len(entries)} files in {workspace}; {report['changed_file_count']} files adapted.")
    print(f"Change log: {workspace / LOG_NAME}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, SyntaxError) as error:
        print(f"Preparation refused: {error}", file=sys.stderr)
        raise SystemExit(2) from error
