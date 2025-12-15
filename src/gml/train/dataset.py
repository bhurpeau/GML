import subprocess
import tempfile
from pathlib import Path
import torch


class DeptGraphDataset(torch.utils.data.Dataset):
    def __init__(self, deps, s3_root):
        self.deps = deps
        self.s3_root = s3_root

    def __len__(self):
        return len(self.deps)

    def __getitem__(self, idx):
        dep = self.deps[idx]
        return load_graph_from_s3(f"{self.s3_root}/{dep}/graph.pt")


def load_graph_from_s3(s3_path: str):
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "graph.pt"
        subprocess.run(["mc", "cp", s3_path, str(local)], check=True)
        return torch.load(local, map_location="cpu")
