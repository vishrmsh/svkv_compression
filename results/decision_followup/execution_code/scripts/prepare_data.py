import argparse
import json
from pathlib import Path
from transformers import AutoTokenizer
from svkv.tasks import build_cases

p = argparse.ArgumentParser()
p.add_argument("--model", default="models/Qwen2.5-7B-Instruct-4bit")
p.add_argument("--output", default="data/pilot_cases.jsonl")
p.add_argument("--longbench-n", type=int, default=6)
a = p.parse_args()
t = AutoTokenizer.from_pretrained(a.model, local_files_only=True)
cases = build_cases(t, "data/longbench/data.zip", longbench_n=a.longbench_n)
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
Path(a.output).write_text("".join(json.dumps(x) + "\n" for x in cases))
manifest = [{k:v for k,v in c.items() if k not in ("prefix", "suffix")} | {"prefix_tokens":len(c["prefix"]),"suffix_tokens":len(c["suffix"])} for c in cases]
Path("data/case_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Wrote {len(cases)} cases; prefix lengths {min(len(c['prefix']) for c in cases)}–{max(len(c['prefix']) for c in cases)}")
