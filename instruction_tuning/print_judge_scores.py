import argparse
import json
from pathlib import Path


CALC_SCORE_KEYS = [
    "faithfulness",
    "answer_correctness",
    "relevance",
    "completeness",
]


def infer_model_name(path: Path) -> str:
    name = path.stem
    for suffix in (
        "_mlx_results_judged",
        "_results_judged",
        "_judged",
    ):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def load_summary(path: Path) -> dict:
    rows = []
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if "scores" in record:
                rows.append(record["scores"])

    if not rows:
        raise ValueError(f"No scored rows found in {path}")

    averages = {
        key: sum(row[key] for row in rows) / len(rows) for key in CALC_SCORE_KEYS
    }
    averages["overall"] = sum(averages[key] for key in CALC_SCORE_KEYS) / len(
        CALC_SCORE_KEYS
    )

    return {
        "model": infer_model_name(path),
        "file": path.name,
        "count": len(rows),
        **averages,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print aggregate judge scores across instruction_tuning eval files."
    )
    parser.add_argument(
        "--eval-dir",
        type=Path,
        default=Path("instruction_tuning/evals"),
        help="Directory containing *_judged.jsonl files.",
    )
    args = parser.parse_args()

    files = sorted(args.eval_dir.glob("*_judged.jsonl"))
    if not files:
        raise SystemExit(f"No *_judged.jsonl files found in {args.eval_dir}")

    summaries = [load_summary(path) for path in files]
    summaries.sort(key=lambda row: row["overall"], reverse=True)

    headers = ["model", "count", "overall", *CALC_SCORE_KEYS, "file"]
    rows = []
    for summary in summaries:
        rows.append(
            [
                summary["model"],
                str(summary["count"]),
                f'{summary["overall"]:.2f}',
                *(f"{summary[key]:.2f}" for key in CALC_SCORE_KEYS),
                summary["file"],
            ]
        )

    widths = [
        max(len(header), *(len(row[i]) for row in rows)) for i, header in enumerate(headers)
    ]

    def format_row(values: list[str]) -> str:
        return "  ".join(value.ljust(widths[i]) for i, value in enumerate(values))

    print(format_row(headers))
    print(format_row(["-" * width for width in widths]))
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()
