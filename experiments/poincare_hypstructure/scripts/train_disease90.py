#!/usr/bin/env python3

import argparse
import json
import logging
import os
import shutil
import sys
import timeit
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA_TSV,
    DEFAULT_RELATIONS_CSV,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_TRAIN_LOG,
    depth_radius_from_checkpoint,
    ensure_poincare_reference_built,
    import_hype_modules,
    normalize_object_ids,
    parse_args_with_defaults,
    read_metadata_tsv,
    read_relations_csv,
    write_json,
)
from hybrid_losses import GlobalHierarchyCPCCLoss, RadialOrderLoss


def selection_score(metric: str, reconstruction_map: float, depth_spearman: float, total_loss: float) -> float:
    if metric == "reconstruction_map":
        return reconstruction_map
    if metric == "depth_spearman":
        return depth_spearman
    if metric == "combined":
        return reconstruction_map + depth_spearman
    if metric == "negative_loss":
        return -total_loss
    raise ValueError(f"Unknown selection metric: {metric}")


def evaluate_checkpoint(
    checkpoint_path: Path,
    config: dict[str, object],
    object_count: int,
    adjacency: dict[int, set[int]],
    metadata_rows: list[dict[str, str]],
):
    build_model, _, _, _, _, eval_reconstruction = import_hype_modules()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config, object_count)
    model.load_state_dict(checkpoint["model"])
    mean_rank, map_rank = eval_reconstruction(adjacency, model)
    depth_metrics = depth_radius_from_checkpoint(checkpoint_path, metadata_rows)
    return {
        "epoch": checkpoint["epoch"],
        "mean_rank": float(mean_rank),
        "map_rank": float(map_rank),
        "depth_radius_pearson": depth_metrics["pearson"],
        "depth_radius_spearman": depth_metrics["spearman"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease-90 Poincare + HypStructure regularized embeddings")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--negs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--burnin", type=int, default=20)
    parser.add_argument("--dampening", type=float, default=0.75)
    parser.add_argument("--ndproc", type=int, default=4)
    parser.add_argument("--eval-each", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--cpcc-weight", type=float, default=0.05)
    parser.add_argument("--radial-weight", type=float, default=0.01)
    parser.add_argument("--radial-margin", type=float, default=0.02)
    parser.add_argument("--cpcc-min-group-size", type=int, default=2)
    parser.add_argument(
        "--selection-metric",
        choices=["combined", "reconstruction_map", "depth_spearman", "negative_loss"],
        default="combined",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parse_args_with_defaults(parser)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Missing relations csv: {args.dataset}")
    if not args.metadata_tsv.exists():
        raise FileNotFoundError(f"Missing metadata tsv: {args.metadata_tsv}")

    relations = read_relations_csv(args.dataset)
    if not relations:
        raise ValueError(f"Relations csv is empty: {args.dataset}")
    metadata_rows = read_metadata_tsv(args.metadata_tsv)

    ensure_poincare_reference_built()
    build_model, LocalCheckpoint, load_edge_list, BatchedDataset, RiemannianSGD, _ = import_hype_modules()

    th = torch
    th.manual_seed(42)
    np.random.seed(42)
    th.set_default_tensor_type("torch.DoubleTensor")
    device = th.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")

    log_level = logging.INFO
    logging.basicConfig(level=log_level, format="%(message)s", stream=sys.stdout)
    log = logging.getLogger("disease90_hyp")

    idx, objects, weights = load_edge_list(str(args.dataset), symmetrize=False)
    normalized_objects = normalize_object_ids(list(objects))
    data = BatchedDataset(
        idx,
        objects,
        weights,
        args.negs,
        args.batchsize,
        args.ndproc,
        burnin=(args.burnin > 0),
        sample_dampening=args.dampening,
    )

    config = vars(args).copy()
    config["manifold"] = "poincare"
    config["model"] = "distance"
    config["margin"] = 0.1
    config["sparse"] = False

    model = build_model(config, len(objects)).to(device)
    optimizer = RiemannianSGD(model.optim_params(), lr=args.lr)
    cpcc_loss_fn = GlobalHierarchyCPCCLoss(
        metadata_rows,
        normalized_objects,
        min_group_size=args.cpcc_min_group_size,
    ).to(device)
    radial_loss_fn = RadialOrderLoss(metadata_rows, normalized_objects, margin=args.radial_margin).to(device)

    checkpoint = LocalCheckpoint(
        str(args.checkpoint),
        include_in_all={"conf": config, "objects": normalized_objects},
        start_fresh=args.fresh,
    )
    state = checkpoint.initialize({"epoch": 0, "model": model.state_dict()})
    model.load_state_dict(state["model"])
    epoch_start = state["epoch"]

    node_to_index = {node_id: index for index, node_id in enumerate(normalized_objects)}
    adjacency = {}
    for child_id, ancestor_id, _ in relations:
        adjacency.setdefault(node_to_index[child_id], set()).add(node_to_index[ancestor_id])

    config_payload = {
        "dataset": str(args.dataset),
        "metadata_tsv": str(args.metadata_tsv),
        "checkpoint": str(args.checkpoint),
        "hyperparameters": {
            "dim": args.dim,
            "epochs": args.epochs,
            "lr": args.lr,
            "negs": args.negs,
            "batchsize": args.batchsize,
            "burnin": args.burnin,
            "dampening": args.dampening,
            "ndproc": args.ndproc,
            "eval_each": args.eval_each,
            "cpcc_weight": args.cpcc_weight,
            "radial_weight": args.radial_weight,
            "radial_margin": args.radial_margin,
            "cpcc_min_group_size": args.cpcc_min_group_size,
            "selection_metric": args.selection_metric,
            "gpu": args.gpu,
        },
    }
    write_json(args.train_config, config_payload)

    best_score = None
    best_payload = None
    env_log_dir = args.log.parent / "mplconfig"
    env_log_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(env_log_dir))

    with args.log.open("w", encoding="utf-8") as log_handle:
        def emit(message: str) -> None:
            print(message)
            log_handle.write(message + "\n")
            log_handle.flush()

        emit(f"json_conf: {json.dumps(config_payload)}")

        for epoch in range(epoch_start, args.epochs):
            data.burnin = epoch < args.burnin
            lr = args.lr * 0.01 if data.burnin else args.lr
            epoch_edge = []
            epoch_cpcc = []
            epoch_radial = []
            epoch_total = []
            start_time = timeit.default_timer()
            iterator = tqdm(data, disable=args.quiet)
            for inputs, targets in iterator:
                inputs = inputs.to(device)
                targets = targets.to(device)
                optimizer.zero_grad()
                preds = model(inputs)
                loss_edge = model.loss(preds, targets, size_average=True)
                loss_cpcc = cpcc_loss_fn(model.lt.weight)
                loss_radial = radial_loss_fn(model.lt.weight)
                total_loss = loss_edge + args.cpcc_weight * loss_cpcc + args.radial_weight * loss_radial
                total_loss.backward()
                optimizer.step(lr=lr)
                model.manifold.normalize(model.lt.weight.data)
                epoch_edge.append(float(loss_edge.detach().cpu()))
                epoch_cpcc.append(float(loss_cpcc.detach().cpu()))
                epoch_radial.append(float(loss_radial.detach().cpu()))
                epoch_total.append(float(total_loss.detach().cpu()))

            elapsed = timeit.default_timer() - start_time
            stats = {
                "epoch": epoch,
                "elapsed": elapsed,
                "edge_loss": float(np.mean(epoch_edge)),
                "cpcc_loss": float(np.mean(epoch_cpcc)),
                "radial_loss": float(np.mean(epoch_radial)),
                "total_loss": float(np.mean(epoch_total)),
            }

            checkpoint_path = Path(f"{args.checkpoint}.{epoch}")
            checkpoint.path = str(checkpoint_path)
            checkpoint.save(
                {
                    "model": model.state_dict(),
                    "embeddings": model.lt.weight.data.detach().cpu(),
                    "epoch": epoch,
                    "model_type": "distance_cpcc_radial",
                }
            )

            if epoch % args.eval_each == args.eval_each - 1:
                eval_stats = evaluate_checkpoint(checkpoint_path, config, len(objects), adjacency, metadata_rows)
                score = selection_score(
                    args.selection_metric,
                    eval_stats["map_rank"],
                    eval_stats["depth_radius_spearman"],
                    stats["total_loss"],
                )
                eval_stats["selection_metric"] = args.selection_metric
                eval_stats["selection_score"] = score
                stats.update(eval_stats)
                is_best = best_score is None or score > best_score
                stats["best"] = bool(is_best)
                if is_best:
                    best_score = score
                    best_payload = stats.copy()
                    shutil.copyfile(checkpoint_path, f"{args.checkpoint}.best")
                    shutil.copyfile(checkpoint_path, args.checkpoint)
            emit(f"json_stats: {json.dumps(stats)}")

    if not Path(f"{args.checkpoint}.best").exists():
        snapshots = sorted(
            args.checkpoint.parent.glob(f"{args.checkpoint.name}.*"),
            key=lambda path: int(path.name.rsplit(".", 1)[-1]) if path.name.rsplit(".", 1)[-1].isdigit() else -1,
        )
        if snapshots:
            shutil.copyfile(snapshots[-1], f"{args.checkpoint}.best")
            shutil.copyfile(snapshots[-1], args.checkpoint)
    if best_payload is not None:
        print(json.dumps(best_payload))


if __name__ == "__main__":
    main()
