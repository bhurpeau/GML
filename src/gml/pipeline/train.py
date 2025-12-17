# gml/pipeline/train.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from gml.train.dataset import DeptGraphDataset
from gml.model.hetero import HeteroGNN
from gml.model.heads import TripletHeads
from gml.model.dmon3p import DMoN3P
from gml.train.train_tripartite import train_dmon3p_multidep

# from gml.io.paths import ensure_local_dir  # si tu as un helper, sinon supprime
from gml.io.duckdb_s3 import s3_put_file

# clés de relations utilisées par la loss (adapte selon ton projet)
XY_KEY = ("adresse", "localise", "bâtiment")
YZ_KEY = ("bâtiment", "appartient", "parcelle")


def assert_consistent_dims(dataset):
    ref = dataset[0]
    ref_dims = {
        "adresse": ref["data"]["adresse"].x.size(1),
        "bâtiment": ref["data"]["bâtiment"].x.size(1),
        "parcelle": ref["data"]["parcelle"].x.size(1),
    }
    for i in range(1, len(dataset)):
        g = dataset[i]
        dims = {
            "adresse": g["data"]["adresse"].x.size(1),
            "bâtiment": g["data"]["bâtiment"].x.size(1),
            "parcelle": g["data"]["parcelle"].x.size(1),
        }
        if dims != ref_dims:
            raise RuntimeError(
                f"Incohérence x_dims sur dep={dataset.deps[i]}: {dims} vs ref {ref_dims}"
            )
    return ref_dims


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--deps", nargs="+", required=True)
    p.add_argument("--s3_root", required=True, help="ex: s3://bhurpeau/WP2/graphs")
    p.add_argument("--out", required=True, help="local out dir, ex: data/models/run1")

    # modèle
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--num_layers", type=int, default=2)

    # DMoN / heads
    p.add_argument("--L", type=int, default=64)
    p.add_argument("--M", type=int, default=64)
    p.add_argument("--N", type=int, default=64)
    p.add_argument("--beta", type=float, default=8.0)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--entropy_weight", type=float, default=5e-4)
    p.add_argument("--lambda_collapse", type=float, default=2e-2)
    p.add_argument("--m_chunk", type=int, default=256)

    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--device", default="cuda")

    # schedules/pruning
    p.add_argument("--anneal_step", type=int, default=10)
    p.add_argument("--anneal_delay_epoch", type=int, default=0)
    p.add_argument("--prune_every", type=int, default=20)
    p.add_argument("--prune_delay_epoch", type=int, default=80)

    # où écrire
    p.add_argument("--save_s3", action="store_true")
    p.add_argument("--s3_ckpt_root", default=None, help="ex: s3://.../models/runX")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Dataset/Loader
    dataset = DeptGraphDataset(args.deps, args.s3_root)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        collate_fn=lambda batch: batch[0],
    )
    ref_dims = assert_consistent_dims(dataset)
    print("OK dims:", ref_dims)

    # 2) Init modèle depuis un graphe “référence”
    sample = dataset[0]
    data0 = sample["data"]
    metadata = data0.metadata()
    node_feature_sizes = {nt: data0[nt].x.size(1) for nt in data0.node_types}

    # edge_feature_size : si ton HeteroGNN en a besoin
    edge_feature_size = 1
    for et in data0.edge_types:
        ea = getattr(data0[et], "edge_attr", None)
        if ea is not None:
            edge_feature_size = ea.size(1) if ea.dim() == 2 else 1
            break

    model = HeteroGNN(
        hidden_channels=args.hidden,
        out_channels=args.emb_dim,
        num_layers=args.num_layers,
        metadata=metadata,
        node_feature_sizes=node_feature_sizes,
        edge_feature_size=edge_feature_size,
    ).to(device)

    heads = TripletHeads(dim=args.emb_dim, L=args.L, M=args.M, N=args.N).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(heads.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # 3) Factory criterion (dépend des tailles par dep)
    def make_criterion(data):
        X = data["adresse"].x.size(0)
        Y = data["bâtiment"].x.size(0)
        Z = data["parcelle"].x.size(0)
        return DMoN3P(
            num_X=X,
            num_Y=Y,
            num_Z=Z,
            L=args.L,
            M=args.M,
            N=args.N,
            beta=args.beta,
            gamma=args.gamma,
            entropy_weight=args.entropy_weight,
            lambda_X=args.lambda_collapse,
            lambda_Y=args.lambda_collapse,
            lambda_Z=args.lambda_collapse,
            m_chunk=args.m_chunk,
        )

    # 4) Train multi-dep
    train_dmon3p_multidep(
        model=model,
        heads=heads,
        optimizer=optimizer,
        loader=loader,
        make_criterion_fn=make_criterion,
        XY_KEY=XY_KEY,
        YZ_KEY=YZ_KEY,
        epochs=args.epochs,
        device=device,
        schedule_beta=(2.0, args.beta, args.anneal_step),
        schedule_gamma=(1.0, args.gamma, args.anneal_step),
        anneal_delay_epoch=args.anneal_delay_epoch,
        prune_every=args.prune_every,
        prune_delay_epoch=args.prune_delay_epoch,
        m_chunk=args.m_chunk,
        use_amp=args.use_amp,
        trial=None,
    )

    # 5) Sauvegarde checkpoint + meta
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = out_dir / f"ckpt_{run_id}.pt"
    meta_path = out_dir / f"meta_{run_id}.json"

    ckpt = {
        "model": model.state_dict(),
        "heads": heads.state_dict(),
        "args": vars(args),
        "metadata": metadata,
        "node_feature_sizes": node_feature_sizes,
        "edge_feature_size": edge_feature_size,
        "run_id": run_id,
    }
    torch.save(ckpt, ckpt_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {"run_id": run_id, "deps": args.deps, "args": vars(args)},
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[OK] checkpoint: {ckpt_path}")
    print(f"[OK] meta      : {meta_path}")

    if args.save_s3:
        if not args.s3_ckpt_root:
            raise ValueError("--save_s3 nécessite --s3_ckpt_root")
        s3_put_file(str(ckpt_path), f"{args.s3_ckpt_root}/{ckpt_path.name}")
        s3_put_file(str(meta_path), f"{args.s3_ckpt_root}/{meta_path.name}")
        print(f"[OK] push S3 -> {args.s3_ckpt_root}")


if __name__ == "__main__":
    main()
