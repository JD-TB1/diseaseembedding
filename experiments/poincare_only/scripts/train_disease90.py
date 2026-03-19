#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_RELATIONS_CSV,
    DEFAULT_TRAIN_CONFIG,
    DEFAULT_TRAIN_LOG,
    build_train_command,
    ensure_poincare_reference_built,
    parse_args_with_defaults,
    read_relations_csv,
    write_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease-90 Poincare embeddings")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-config", type=Path, default=DEFAULT_TRAIN_CONFIG)
    parser.add_argument("--log", type=Path, default=DEFAULT_TRAIN_LOG)
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--negs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=256)
    parser.add_argument("--burnin", type=int, default=20)
    parser.add_argument("--dampening", type=float, default=0.75)
    parser.add_argument("--ndproc", type=int, default=4)
    parser.add_argument("--eval-each", type=int, default=5)
    parser.add_argument("--train-threads", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--sparse", action="store_true", default=True)
    parser.add_argument("--no-sparse", dest="sparse", action="store_false")
    args = parse_args_with_defaults(parser)

    if not args.dataset.exists():
        raise FileNotFoundError(f"Missing relations csv: {args.dataset}")
    relations = read_relations_csv(args.dataset)
    if not relations:
        raise ValueError(f"Relations csv is empty: {args.dataset}")

    ensure_poincare_reference_built()
    command = build_train_command(
        dataset_path=args.dataset,
        checkpoint_path=args.checkpoint,
        dim=args.dim,
        epochs=args.epochs,
        lr=args.lr,
        negs=args.negs,
        batchsize=args.batchsize,
        burnin=args.burnin,
        dampening=args.dampening,
        ndproc=args.ndproc,
        eval_each=args.eval_each,
        train_threads=args.train_threads,
        gpu=args.gpu,
        sparse=args.sparse,
        fresh=args.fresh,
    )
    payload = {
        "command": command,
        "command_shell": " ".join(shlex.quote(part) for part in command),
        "dataset": str(args.dataset),
        "checkpoint": str(args.checkpoint),
        "relation_count": len(relations),
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
            "train_threads": args.train_threads,
            "gpu": args.gpu,
            "sparse": args.sparse,
            "fresh": args.fresh,
        },
    }
    write_json(args.train_config, payload)

    env = os.environ.copy()
    mpl_dir = args.log.parent / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)

    with args.log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)
        returncode = process.wait()
    if returncode != 0:
        raise SystemExit(returncode)

    best_path = Path(f"{args.checkpoint}.best")
    if not best_path.exists():
        epoch_snapshots = sorted(
            args.checkpoint.parent.glob(f"{args.checkpoint.name}.*"),
            key=lambda path: int(path.name.rsplit(".", 1)[-1]) if path.name.rsplit(".", 1)[-1].isdigit() else -1,
        )
        if epoch_snapshots:
            latest = epoch_snapshots[-1]
            shutil.copyfile(latest, args.checkpoint)
            shutil.copyfile(latest, best_path)
    if not best_path.exists():
        raise FileNotFoundError(f"Training did not create best checkpoint: {best_path}")
    print(json.dumps({"checkpoint": str(args.checkpoint), "best_checkpoint": str(best_path)}))


if __name__ == "__main__":
    main()
