import subprocess
import tempfile
from pathlib import Path
import torch
import json


class DeptGraphDataset(torch.utils.data.Dataset):
    def __init__(self, deps, s3_root):
        self.deps = deps
        self.s3_root = s3_root.rstrip("/")

    def __len__(self):
        return len(self.deps)

    def __getitem__(self, idx):
        dep = self.deps[idx]
        base = f"{self.s3_root}/{dep}"

        graph = load_torch_from_s3(f"{base}/graph.pt")
        bat_map = load_json_from_s3(f"{base}/bat_map.json")

        # optionnels mais très utiles
        # parcelle_map = try_load_json_from_s3(f"{base}/parcelle_map.json")
        # adresse_map = try_load_json_from_s3(f"{base}/adresse_map.json")

        return {
            "dep": dep,
            "data": graph,
            "bat_map": bat_map,
            #    "parcelle_map": parcelle_map,
            #    "adresse_map": adresse_map,
        }


def load_torch_from_s3(s3_path: str):
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / Path(s3_path).name
        s3_path = s3_path.replace("s3://", "s3/")
        print(s3_path)
        subprocess.run(["mc", "cp", s3_path, str(local)], check=True)
        return torch.load(local, map_location="cpu", weights_only=False)


def load_json_from_s3(s3_path: str):
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / Path(s3_path).name
        s3_path = s3_path.replace("s3://", "s3/")
        print(s3_path)
        subprocess.run(["mc", "cp", s3_path, str(local)], check=True)
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)


def try_load_json_from_s3(s3_path: str):
    try:
        return load_json_from_s3(s3_path)
    except Exception:
        return None
