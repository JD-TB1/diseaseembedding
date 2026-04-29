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
    PROJECT_DIR,
    depth_radius_from_checkpoint,
    ensure_poincare_reference_built,
    import_hype_modules,
    normalize_object_ids,
    parse_args_with_defaults,
    read_metadata_tsv,
    read_relations_csv,
    write_json,
)
from hybrid_losses import (
    BranchAngularSeparationLoss,
    BranchContrastiveMarginLoss,
    BranchTeacherLayoutLoss,
    DepthBandLoss,
    DepthQuantileMarginLoss,
    GlobalHierarchyCPCCLoss,
    RadialOrderLoss,
)


GEOMETRY_RAMP_START = 50
GEOMETRY_RAMP_END = 150


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


def geometry_schedule_scale(epoch: int, schedule: str) -> float:
    if schedule == "off":
        return 0.0
    if schedule == "constant":
        return 1.0
    if schedule != "ramp":
        raise ValueError(f"Unknown geometry schedule: {schedule}")
    if epoch < GEOMETRY_RAMP_START:
        return 0.0
    if epoch >= GEOMETRY_RAMP_END:
        return 1.0
    return float((epoch - GEOMETRY_RAMP_START) / (GEOMETRY_RAMP_END - GEOMETRY_RAMP_START))


def checkpoint_epoch(path: Path) -> int:
    suffix = path.name.rsplit(".", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def latest_numbered_checkpoint(prefix: Path) -> Path | None:
    snapshots = sorted(
        [path for path in prefix.parent.glob(f"{prefix.name}.*") if checkpoint_epoch(path) >= 0],
        key=checkpoint_epoch,
    )
    return snapshots[-1] if snapshots else None


def direct_poincare_init_checkpoint() -> Path | None:
    direct_prefix = PROJECT_DIR / "experiments" / "poincare_only" / "results" / "disease90" / "disease90_embeddings_direct.pth"
    candidates = [Path(f"{direct_prefix}.best"), direct_prefix]
    latest = latest_numbered_checkpoint(direct_prefix)
    if latest is not None:
        candidates.append(latest)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def current_hybrid_init_checkpoint() -> Path | None:
    geometry_stage0_best = (
        PROJECT_DIR
        / "experiments"
        / "poincare_hypstructure"
        / "tuning"
        / "geometry_separation"
        / "stage0"
        / "current_hybrid"
        / "current_hybrid.offline_best.pth"
    )
    hybrid_prefix = DEFAULT_CHECKPOINT
    candidates = [geometry_stage0_best, Path(f"{hybrid_prefix}.offline_best"), Path(f"{hybrid_prefix}.best"), hybrid_prefix]
    latest = latest_numbered_checkpoint(hybrid_prefix)
    if latest is not None:
        candidates.append(latest)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_init_checkpoint(explicit_checkpoint: Path | None, init_source: str) -> tuple[Path | None, bool, str]:
    if explicit_checkpoint is not None:
        if not explicit_checkpoint.exists():
            raise FileNotFoundError(f"Requested init checkpoint does not exist: {explicit_checkpoint}")
        return explicit_checkpoint, True, "explicit"

    if init_source == "none":
        return None, False, "none"
    if init_source == "direct-poincare":
        return direct_poincare_init_checkpoint(), False, "direct-poincare"
    if init_source != "current-hybrid":
        raise ValueError(f"Unknown init source: {init_source}")

    current_hybrid = current_hybrid_init_checkpoint()
    if current_hybrid is not None:
        return current_hybrid, False, "current-hybrid"
    return direct_poincare_init_checkpoint(), False, "direct-poincare-fallback"


def load_aligned_teacher_embeddings(
    teacher_checkpoint: Path,
    normalized_objects: list[str],
    device: torch.device,
) -> torch.Tensor:
    checkpoint = torch.load(teacher_checkpoint, map_location="cpu", weights_only=False)
    embeddings = checkpoint.get("embeddings")
    objects = checkpoint.get("objects")
    if embeddings is None or objects is None:
        raise ValueError(f"Teacher checkpoint is missing embeddings/objects: {teacher_checkpoint}")
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    source_objects = normalize_object_ids(list(objects))
    source_index = {node_id: index for index, node_id in enumerate(source_objects)}
    missing = [node_id for node_id in normalized_objects if node_id not in source_index]
    if missing:
        raise ValueError(f"Teacher checkpoint is missing {len(missing)} objects, example: {missing[:5]}")
    aligned = torch.stack([embeddings[source_index[node_id]] for node_id in normalized_objects], dim=0)
    return aligned.to(device=device, dtype=torch.double)


def initialize_from_checkpoint(
    model,
    init_checkpoint: Path | None,
    normalized_objects: list[str],
    explicit: bool,
) -> dict[str, object]:
    if init_checkpoint is None:
        return {"status": "scratch", "path": None, "reason": "no_init_checkpoint_found"}

    checkpoint = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
    embeddings = checkpoint.get("embeddings")
    objects = checkpoint.get("objects")
    if embeddings is None or objects is None:
        if explicit:
            raise ValueError(f"Init checkpoint is missing embeddings/objects: {init_checkpoint}")
        return {"status": "scratch", "path": str(init_checkpoint), "reason": "missing_embeddings_or_objects"}

    source_objects = normalize_object_ids(list(objects))
    source_index = {node_id: index for index, node_id in enumerate(source_objects)}
    missing = [node_id for node_id in normalized_objects if node_id not in source_index]
    if missing:
        if explicit:
            raise ValueError(f"Init checkpoint is missing {len(missing)} objects, example: {missing[:5]}")
        return {"status": "scratch", "path": str(init_checkpoint), "reason": "object_mismatch"}

    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    if embeddings.size(1) != model.lt.weight.data.size(1):
        if explicit:
            raise ValueError(
                f"Init dim {embeddings.size(1)} does not match model dim {model.lt.weight.data.size(1)}"
            )
        return {"status": "scratch", "path": str(init_checkpoint), "reason": "dimension_mismatch"}

    aligned = torch.stack([embeddings[source_index[node_id]] for node_id in normalized_objects], dim=0)
    model.lt.weight.data.copy_(aligned.to(device=model.lt.weight.device, dtype=model.lt.weight.dtype))
    model.manifold.normalize(model.lt.weight.data)
    return {
        "status": "initialized",
        "path": str(init_checkpoint),
        "reason": "ok",
        "source_epoch": checkpoint.get("epoch"),
        "explicit": explicit,
    }


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
    parser.add_argument("--depth-band-weight", type=float, default=0.0)
    parser.add_argument("--depth-quantile-weight", type=float, default=0.0)
    parser.add_argument("--depth-quantile-margin", type=float, default=0.001)
    parser.add_argument("--branch-weight", type=float, default=0.0)
    parser.add_argument("--branch-cos-margin", type=float, default=0.2)
    parser.add_argument("--branch-teacher-weight", type=float, default=0.0)
    parser.add_argument("--branch-teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--branch-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--branch-contrastive-margin", type=float, default=0.02)
    parser.add_argument("--branch-contrastive-hard-k", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-source",
        choices=["current-hybrid", "direct-poincare", "none"],
        default="current-hybrid",
    )
    parser.add_argument("--geometry-schedule", choices=["ramp", "constant", "off"], default="ramp")
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
    depth_band_loss_fn = DepthBandLoss(metadata_rows, normalized_objects).to(device)
    depth_quantile_loss_fn = DepthQuantileMarginLoss(
        metadata_rows,
        normalized_objects,
        margin=args.depth_quantile_margin,
    ).to(device)
    branch_loss_fn = BranchAngularSeparationLoss(
        metadata_rows,
        normalized_objects,
        cos_margin=args.branch_cos_margin,
    ).to(device)
    branch_contrastive_loss_fn = BranchContrastiveMarginLoss(
        metadata_rows,
        normalized_objects,
        margin=args.branch_contrastive_margin,
        hard_negative_k=args.branch_contrastive_hard_k,
    ).to(device)

    checkpoint = LocalCheckpoint(
        str(args.checkpoint),
        include_in_all={"conf": config, "objects": normalized_objects},
        start_fresh=args.fresh,
    )
    state = checkpoint.initialize({"epoch": 0, "model": model.state_dict()})
    model.load_state_dict(state["model"])
    epoch_start = state["epoch"]
    init_candidate, init_explicit, init_source = resolve_init_checkpoint(args.init_checkpoint, args.init_source)
    init_payload = {"status": "not_applied", "path": None, "reason": "resuming_existing_checkpoint"}
    if epoch_start == 0 and (args.fresh or args.init_checkpoint is not None or not args.checkpoint.exists()):
        init_payload = initialize_from_checkpoint(model, init_candidate, normalized_objects, init_explicit)
    init_payload["source"] = init_source

    teacher_checkpoint = args.branch_teacher_checkpoint
    if teacher_checkpoint is None and args.branch_teacher_weight > 0.0:
        teacher_checkpoint = current_hybrid_init_checkpoint()
    if teacher_checkpoint is None and args.branch_teacher_weight > 0.0:
        raise FileNotFoundError("Branch teacher loss requested, but no current-hybrid teacher checkpoint was found")
    branch_teacher_payload = {"status": "disabled", "path": None}
    if teacher_checkpoint is not None and args.branch_teacher_weight > 0.0:
        teacher_embeddings = load_aligned_teacher_embeddings(teacher_checkpoint, normalized_objects, device)
        branch_teacher_loss_fn = BranchTeacherLayoutLoss(metadata_rows, normalized_objects, teacher_embeddings).to(device)
        branch_teacher_payload = {"status": "loaded", "path": str(teacher_checkpoint)}
    else:
        branch_teacher_loss_fn = None

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
            "depth_band_weight": args.depth_band_weight,
            "depth_quantile_weight": args.depth_quantile_weight,
            "depth_quantile_margin": args.depth_quantile_margin,
            "branch_weight": args.branch_weight,
            "branch_cos_margin": args.branch_cos_margin,
            "branch_teacher_weight": args.branch_teacher_weight,
            "branch_teacher_checkpoint": str(args.branch_teacher_checkpoint) if args.branch_teacher_checkpoint else None,
            "branch_contrastive_weight": args.branch_contrastive_weight,
            "branch_contrastive_margin": args.branch_contrastive_margin,
            "branch_contrastive_hard_k": args.branch_contrastive_hard_k,
            "init_source": args.init_source,
            "geometry_schedule": args.geometry_schedule,
            "selection_metric": args.selection_metric,
            "gpu": args.gpu,
        },
        "init_checkpoint": init_payload,
        "branch_teacher": branch_teacher_payload,
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
            epoch_depth_band = []
            epoch_depth_quantile = []
            epoch_branch = []
            epoch_branch_teacher = []
            epoch_branch_contrastive = []
            epoch_total = []
            geometry_scale = geometry_schedule_scale(epoch, args.geometry_schedule)
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
                loss_depth_band = depth_band_loss_fn(model.lt.weight)
                loss_depth_quantile = depth_quantile_loss_fn(model.lt.weight)
                loss_branch = branch_loss_fn(model.lt.weight)
                loss_branch_teacher = (
                    branch_teacher_loss_fn(model.lt.weight)
                    if branch_teacher_loss_fn is not None
                    else model.lt.weight.new_tensor(0.0)
                )
                loss_branch_contrastive = branch_contrastive_loss_fn(model.lt.weight)
                total_loss = (
                    loss_edge
                    + geometry_scale * args.cpcc_weight * loss_cpcc
                    + geometry_scale * args.radial_weight * loss_radial
                    + geometry_scale * args.depth_band_weight * loss_depth_band
                    + geometry_scale * args.depth_quantile_weight * loss_depth_quantile
                    + geometry_scale * args.branch_weight * loss_branch
                    + geometry_scale * args.branch_teacher_weight * loss_branch_teacher
                    + geometry_scale * args.branch_contrastive_weight * loss_branch_contrastive
                )
                total_loss.backward()
                optimizer.step(lr=lr)
                model.manifold.normalize(model.lt.weight.data)
                epoch_edge.append(float(loss_edge.detach().cpu()))
                epoch_cpcc.append(float(loss_cpcc.detach().cpu()))
                epoch_radial.append(float(loss_radial.detach().cpu()))
                epoch_depth_band.append(float(loss_depth_band.detach().cpu()))
                epoch_depth_quantile.append(float(loss_depth_quantile.detach().cpu()))
                epoch_branch.append(float(loss_branch.detach().cpu()))
                epoch_branch_teacher.append(float(loss_branch_teacher.detach().cpu()))
                epoch_branch_contrastive.append(float(loss_branch_contrastive.detach().cpu()))
                epoch_total.append(float(total_loss.detach().cpu()))

            elapsed = timeit.default_timer() - start_time
            stats = {
                "epoch": epoch,
                "elapsed": elapsed,
                "edge_loss": float(np.mean(epoch_edge)),
                "cpcc_loss": float(np.mean(epoch_cpcc)),
                "radial_loss": float(np.mean(epoch_radial)),
                "depth_band_loss": float(np.mean(epoch_depth_band)),
                "depth_quantile_loss": float(np.mean(epoch_depth_quantile)),
                "branch_loss": float(np.mean(epoch_branch)),
                "branch_teacher_loss": float(np.mean(epoch_branch_teacher)),
                "branch_contrastive_loss": float(np.mean(epoch_branch_contrastive)),
                "geometry_scale": geometry_scale,
                "total_loss": float(np.mean(epoch_total)),
            }

            checkpoint_path = Path(f"{args.checkpoint}.{epoch}")
            checkpoint.path = str(checkpoint_path)
            checkpoint.save(
                {
                    "model": model.state_dict(),
                    "embeddings": model.lt.weight.data.detach().cpu(),
                    "epoch": epoch,
                    "model_type": "distance_cpcc_radial_depth_branch",
                    "init_checkpoint": init_payload,
                    "branch_teacher": branch_teacher_payload,
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
