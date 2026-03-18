from __future__ import annotations

import argparse
from pathlib import Path

from .data import load_players_df
from .features import prepare_round_classification
from .models import fit_and_evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="nba-draft-success")
    sub = p.add_subparsers(dest="command", required=True)

    train = sub.add_parser("train", help="Train and evaluate a draft-round classifier")
    train.add_argument(
        "--model",
        choices=["knn", "dt", "rf", "mlp"],
        default="rf",
        help="Model family to use",
    )
    train.add_argument(
        "--csv-path",
        type=Path,
        default=None,
        help="Optional path to players.csv (defaults to repo players.csv)",
    )
    train.add_argument("--test-size", type=float, default=0.2)
    train.add_argument("--random-state", type=int, default=42)
    train.add_argument("--na-strategy", choices=["drop", "median"], default="drop")

    return p


def cmd_train(args: argparse.Namespace) -> int:
    df = load_players_df(args.csv_path)
    X, y = prepare_round_classification(df, na_strategy=args.na_strategy)
    _, result = fit_and_evaluate(
        X,
        y,
        model=args.model,
        test_size=args.test_size,
        random_state=args.random_state,
    )

    print(f"Accuracy: {result.accuracy:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred; labels 0/1/2):")
    print(result.confusion)
    print("\nClassification report:")
    print(result.report)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "train":
        return cmd_train(args)

    raise RuntimeError(f"Unhandled command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())

