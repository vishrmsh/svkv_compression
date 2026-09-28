"""Run the fixed 36-case decision-score follow-up with immutable input/source checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


def sha256(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/decision_followup")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    protocol_path = Path("configs/decision_followup.json")
    protocol = json.loads(protocol_path.read_text())
    case_path = Path(protocol["cases"])
    if sha256(case_path) != protocol["case_file_sha256"]:
        raise ValueError("Case file differs from the frozen original pilot")
    cases = [json.loads(line) for line in case_path.read_text().splitlines()]
    cases = [case for case in cases if case["task"] in protocol["task_filter"]]
    if len(cases) != 36 or len({case["id"] for case in cases}) != 36:
        raise ValueError("Expected exactly 36 unique original needle cases")
    original_path = Path("results/pilot/predictions.jsonl")
    original = [json.loads(line) for line in original_path.read_text().splitlines()]
    for case in cases:
        matches = [row for row in original if row["case_id"] == case["id"]]
        if len(matches) != 25 or any(row["prefix_sha256"] != case["prefix_sha256"] for row in matches):
            raise ValueError(f"Original pilot case mismatch: {case['id']}")
    sources = [
        protocol_path, Path("configs/pilot_protocol.json"), Path("requirements-lock.txt"),
        Path("scripts/run_decision_followup.py"), Path("scripts/run_pilot.py"),
        Path("scripts/prepare_data.py"), *sorted(Path("src/svkv").glob("*.py")),
    ]
    manifest = {
        "purpose": protocol["purpose"],
        "source_sha256": {str(path): sha256(path) for path in sources},
        "case_file_sha256": sha256(case_path),
        "original_predictions_sha256": sha256(original_path),
        "asset_manifest_sha256": sha256("data/asset_manifest.json"),
        "case_ids": [case["id"] for case in cases],
        "expected_answers": protocol["expected_answers"],
    }
    out = Path(args.output)
    if out.resolve() == Path("results/pilot").resolve():
        raise ValueError("Never overwrite the original pilot")
    manifest_path = out / "followup_manifest.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Resume requires identical saved inputs and execution sources")
    else:
        if out.exists() and any(out.iterdir()):
            raise ValueError("Output directory is nonempty; use a new directory or unchanged --resume")
        out.mkdir(parents=True, exist_ok=True)
        for path in sources:
            target = out / "execution_code" / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    for path, expected in manifest["source_sha256"].items():
        if sha256(out / "execution_code" / path) != expected:
            raise ValueError(f"Archived execution source changed: {path}")
    command = [
        sys.executable, "scripts/run_pilot.py", "--cases", str(case_path),
        "--output", str(out), "--methods", "svdd_decision", "keydiff", "leverage",
        "--budgets", "0.1", "0.2", "0.5", "--ids", *manifest["case_ids"],
    ]
    print("Frozen decision-score follow-up: 36 cases, 360 answers; no parameter sweep.", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
