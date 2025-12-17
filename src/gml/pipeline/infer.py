# gml/pipeline/infer.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader

from gml.data.dataset import DeptGraphDataset
from gml.model.hetero import HeteroGNN
from gml.model.heads import TripletHeads
from gml.io.s3 import s3_get_file, s3_put_file  # à toi

# mêmes clés que training
XY_KEY = ("adresse", "localise", "bâtiment")
YZ_KEY = ("bâtiment", "appartient", "parcelle")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deps", nargs="+", required=True)
    p.add_argument("--s3_root", required=True, help="ex: s3://.../graphs")
    p.add_argument("--ckpt", required=True, help="local path ou s3://.../ckpt.pt")
    p.add_argument("--out", required=True, help="local out dir")

    p.add_argument("--device", default="cuda")

    p.add_argument("--save_s3", action="store_true")
    p.add_argument("--s3_pred_root", default=None, help="ex: s3://.../predictions/runX")
    return p.parse_args()


def load_checkpoint(ckpt_arg: str, local_dir: Path, device: str):
    local_dir.mkdir(parents=True, exist_ok=True)
    if ckpt_arg.startswith("s3://"):
        local_path = local_dir / Path(ckpt_arg).name
        s3_get_file(ckpt_arg, str(local_path))
    else:
        local_path = Path(ckpt_arg)
    ckpt = torch.load(local_path, map_location=device)
    return ckpt, local_path


@torch.no_grad()
def infer_one_dep(model, heads, data, device: str):
    data = data.to(device)
    x_dict = {k: v.to(device) for k, v in data.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in data.edge_index_dict.items()}

    edge_attr_dict = {}
    for rel in data.edge_types:
        ea = getattr(data[rel], "edge_attr", None)
        edge_attr_dict[rel] = ea.to(device) if ea is not None else None

    h_dict = model(x_dict, edge_index_dict, edge_attr_dict)
    hX, hY, hZ = h_dict["adresse"], h_dict["bâtiment"], h_dict["parcelle"]

    Sx_logits, Sy_logits, Sz_logits, gates = heads(hX, hY, hZ)

    # clusters = argmax (tu peux aussi sauvegarder les probas softmax)
    yX = torch.argmax(Sx_logits, dim=1).cpu().numpy()
    yY = torch.argmax(Sy_logits, dim=1).cpu().numpy()
    yZ = torch.argmax(Sz_logits, dim=1).cpu().numpy()

    return yX, yY, yZ


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, ckpt_path = load_checkpoint(args.ckpt, out_dir / "_ckpt", device)

    # Rebuild architecture identique
    model = HeteroGNN(
        hidden_channels=ckpt["args"]["hidden"],
        out_channels=ckpt["args"]["emb_dim"],
        num_layers=ckpt["args"].get("num_layers", 2),
        metadata=ckpt["metadata"],
        node_feature_sizes=ckpt["node_feature_sizes"],
        edge_feature_size=ckpt.get("edge_feature_size", 1),
    ).to(device)

    heads = TripletHeads(
        dim=ckpt["args"]["emb_dim"],
        L=ckpt["args"]["L"],
        M=ckpt["args"]["M"],
        N=ckpt["args"]["N"],
    ).to(device)

    model.load_state_dict(ckpt["model"])
    heads.load_state_dict(ckpt["heads"])
    model.eval()
    heads.eval()

    # Dataset/Loader
    dataset = DeptGraphDataset(args.deps, args.s3_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    run_id = ckpt.get("run_id", "run")
    for i, batch in enumerate(loader):
        data = batch[0] if isinstance(batch, (list, tuple)) else batch
        dep = args.deps[i]  # batch_size=1, shuffle=False => alignement simple

        yX, yY, yZ = infer_one_dep(model, heads, data, device=device)

        # Ici tu veux rabouter aux IDs (bat_map) pour export.
        # -> Soit DeptGraphDataset fournit bat_map, soit tu le lis sur S3.
        # Placeholder : à adapter.
        # bat_map = load_bat_map(dep)
        # df_bat = pd.DataFrame({"rnb_id": ..., "cluster": yY})

        out_dep_dir = out_dir / dep
        out_dep_dir.mkdir(parents=True, exist_ok=True)

        # Exports minimaux (indices internes)
        pd.DataFrame({"cluster": yX}).to_parquet(
            out_dep_dir / "adresse_clusters.parquet"
        )
        pd.DataFrame({"cluster": yY}).to_parquet(
            out_dep_dir / "batiment_clusters.parquet"
        )
        pd.DataFrame({"cluster": yZ}).to_parquet(
            out_dep_dir / "parcelle_clusters.parquet"
        )

        print(f"[OK] infer dep {dep} -> {out_dep_dir}")

        if args.save_s3:
            if not args.s3_pred_root:
                raise ValueError("--save_s3 nécessite --s3_pred_root")
            s3_put_file(
                str(out_dep_dir / "adresse_clusters.parquet"),
                f"{args.s3_pred_root}/{run_id}/{dep}/adresse_clusters.parquet",
            )
            s3_put_file(
                str(out_dep_dir / "batiment_clusters.parquet"),
                f"{args.s3_pred_root}/{run_id}/{dep}/batiment_clusters.parquet",
            )
            s3_put_file(
                str(out_dep_dir / "parcelle_clusters.parquet"),
                f"{args.s3_pred_root}/{run_id}/{dep}/parcelle_clusters.parquet",
            )

    print("[OK] inference done")


if __name__ == "__main__":
    main()
