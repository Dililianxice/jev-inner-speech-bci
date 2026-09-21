import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits


def benchmark_main():
    from .benchmark import main as run_benchmark

    parser = argparse.ArgumentParser(
        description="Run the frozen forward-block inner-speech benchmark."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Extracted Dryad directory containing exactly four MAT files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Destination for derived tables.",
    )
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run_benchmark(args.data_dir, args.output_dir)


def report_main():
    from .report import main as run_report

    parser = argparse.ArgumentParser(
        description="Regenerate tables and figures from benchmark CSV files."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--figure-dir", type=Path, default=Path("figures"))
    args = parser.parse_args()
    run_report(args.output_dir, args.figure_dir)
