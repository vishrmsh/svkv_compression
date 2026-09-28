"""Create deterministic compressed solver evidence for a manageable Git repo."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

records = []
for source in sorted(Path("results").glob("*/solver_diagnostics.jsonl")):
    target = source.with_suffix(source.suffix + ".gz")
    with source.open("rb") as original, target.open("wb") as output:
        with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as compressed:
            shutil.copyfileobj(original, compressed)
    records.append({"raw_path":str(source), "raw_bytes":source.stat().st_size,
                    "raw_sha256":hashlib.file_digest(source.open("rb"), "sha256").hexdigest(),
                    "compressed_path":str(target), "compressed_bytes":target.stat().st_size,
                    "compressed_sha256":hashlib.file_digest(target.open("rb"), "sha256").hexdigest()})
Path("results/evidence_manifest.json").write_text(json.dumps(records,indent=2)+"\n")
print(json.dumps(records,indent=2))
