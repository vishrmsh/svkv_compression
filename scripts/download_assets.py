"""Download the exact public assets used by the pilot into the workspace."""
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
snapshot_download(
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    revision="c26a38f6a37d0a51b4e9a1eb3026530fa35d9fed",
    local_dir=ROOT / "models/Qwen2.5-7B-Instruct-4bit",
)
hf_hub_download(
    "THUDM/LongBench", "data.zip", repo_type="dataset",
    revision="5e628be450b7e67fb7ae6e201bd6d8f7056f7672",
    local_dir=ROOT / "data/longbench",
)
