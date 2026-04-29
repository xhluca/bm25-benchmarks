"""Command line interface for running BM25 benchmark evals."""

from __future__ import annotations

import argparse
import cProfile
import importlib
import pstats
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable


DEFAULT_DATASET = "fiqa"

AVAILABLE_DATASETS = (
    "trec-covid",
    "nfcorpus",
    "fiqa",
    "arguana",
    "webis-touche2020",
    "quora",
    "scidocs",
    "scifact",
    "cqadupstack",
    "nq",
    "msmarco",
    "hotpotqa",
    "dbpedia-entity",
    "fever",
    "climate-fever",
)


@dataclass(frozen=True)
class Backend:
    name: str
    module: str
    profile_filename: str | None = None


BACKENDS = {
    "bm25s": Backend("bm25s", "benchmark.on_bm25s", "bm25s.prof"),
    "rank-bm25": Backend("rank-bm25", "benchmark.on_rank_bm25", "rankbm25.prof"),
    "bm25-pt": Backend("bm25-pt", "benchmark.on_bm25_pt", "bm25pt.prof"),
    "pyserini": Backend("pyserini", "benchmark.on_pyserini"),
    "elastic": Backend("elastic", "benchmark.on_elastic", "elastic.prof"),
    "pisa": Backend("pisa", "benchmark.on_pisa"),
}

PINNED_RANK_BM25_REQUIREMENT = (
    "rank-bm25 @ "
    "git+https://github.com/dorianbrown/rank_bm25.git"
    "@1abce6cb8bd4a4961f0958391b3eabb749483c01"
)

INSTALL_REQUIREMENTS = {
    "bm25s": [
        "bm25s[core]>=0.3.8",
    ],
    "rank-bm25": [
        "beir",
        "PyStemmer",
        "ujson",
        "numpy",
        PINNED_RANK_BM25_REQUIREMENT,
    ],
    "bm25-pt": [
        "beir",
        "PyStemmer",
        "ujson",
        "bm25-pt",
        "transformers",
    ],
    "pyserini": [
        "beir",
        "ujson",
        "pyserini",
    ],
    "elastic": [
        "beir",
        "ujson",
        "elasticsearch",
    ],
    "pisa": [
        "beir",
        "ujson",
        "pandas",
        "pyterrier_pisa>=0.3.0",
    ],
}

INSTALL_ALIASES = {
    "rank": "rank-bm25",
    "rank_bm25": "rank-bm25",
    "bm25_pt": "bm25-pt",
    "elastic-bm25": "elastic",
}


def _split_csv(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return default

    expanded: list[str] = []
    for value in values:
        expanded.extend(item.strip() for item in value.split(",") if item.strip())

    return expanded or default


def _load_main(backend: Backend) -> Callable[..., Any]:
    module = importlib.import_module(backend.module)
    return module.main


def _run_once(backend: Backend, kwargs: dict[str, Any]) -> None:
    main_fn = _load_main(backend)
    main_fn(**kwargs)


def _run_profiled(backend: Backend, kwargs: dict[str, Any]) -> None:
    if backend.profile_filename is None:
        raise ValueError(f"Profiling is not configured for {backend.name}.")

    main_fn = _load_main(backend)

    profiler = cProfile.Profile()
    profiler.enable()
    main_fn(**kwargs)
    profiler.disable()
    profiler.dump_stats(backend.profile_filename)

    stats = pstats.Stats(backend.profile_filename)
    stats.sort_stats("time").print_stats(50)


def _run_backend(
    backend_name: str,
    datasets: list[str],
    kwargs: dict[str, Any],
    num_runs: int,
    profile: bool,
    dry_run: bool,
    continue_on_error: bool,
) -> int:
    backend = BACKENDS[backend_name]

    if profile and not dry_run and (num_runs > 1 or len(datasets) > 1):
        raise ValueError("Cannot profile multiple datasets or multiple runs.")

    failures = 0
    for dataset in datasets:
        run_kwargs = dict(kwargs)
        run_kwargs["dataset"] = dataset

        for run_idx in range(num_runs):
            label = f"{backend.name} on {dataset}"
            if num_runs > 1:
                label += f" (run {run_idx + 1}/{num_runs})"

            print(f"Running {label}")
            if dry_run:
                print(f"  kwargs: {run_kwargs}")
                continue

            try:
                if profile:
                    _run_profiled(backend, run_kwargs)
                else:
                    _run_once(backend, run_kwargs)
            except Exception as exc:
                failures += 1
                if not continue_on_error:
                    raise
                print(f"Failed {label}: {exc}")

    return failures


def _common_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-d",
        "--dataset",
        dest="datasets",
        action="append",
        help=(
            "Dataset to evaluate. Repeat the flag or pass a comma-separated list "
            f"for multiple datasets. Defaults to {DEFAULT_DATASET}."
        ),
    )
    parser.add_argument(
        "-t",
        "--threads",
        "--n-threads",
        "--n_threads",
        dest="n_threads",
        type=int,
        default=1,
        help="Number of threads/jobs to use where the backend supports it.",
    )
    parser.add_argument(
        "--top-k",
        "--top_k",
        dest="top_k",
        type=int,
        default=1000,
        help="Number of top documents to retrieve.",
    )
    parser.add_argument(
        "--save-dir",
        "--save_dir",
        dest="save_dir",
        default="datasets",
        help="Directory used for downloaded datasets.",
    )
    parser.add_argument(
        "--result-dir",
        "--result_dir",
        dest="result_dir",
        default="results",
        help="Directory used for JSON benchmark outputs.",
    )
    parser.add_argument(
        "--num-runs",
        "--num_runs",
        dest="num_runs",
        type=int,
        default=1,
        help="Number of repeated runs for each selected dataset.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile one run and print the slowest functions.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the backend calls without running downloads or evals.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue evaluating remaining datasets after a backend failure.",
    )
    return parser


def _bm25_params_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 parameter.")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    return parser


def _finalize_common_args(args: argparse.Namespace) -> tuple[list[str], int, bool, bool, bool]:
    datasets = _split_csv(args.datasets, [DEFAULT_DATASET])
    num_runs = args.num_runs
    profile = args.profile
    dry_run = args.dry_run
    continue_on_error = args.continue_on_error

    del args.datasets
    del args.num_runs
    del args.profile
    del args.dry_run
    del args.continue_on_error

    return datasets, num_runs, profile, dry_run, continue_on_error


def run_eval(args: argparse.Namespace) -> int:
    datasets, num_runs, profile, dry_run, continue_on_error = _finalize_common_args(args)
    backend_name = args.backend
    del args.backend

    kwargs = {
        key: value
        for key, value in vars(args).items()
        if key not in {"command", "func"} and value is not None
    }
    failures = _run_backend(
        backend_name=backend_name,
        datasets=datasets,
        kwargs=kwargs,
        num_runs=num_runs,
        profile=profile,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )
    return 1 if failures else 0


def list_models(_: argparse.Namespace) -> int:
    for name in BACKENDS:
        print(name)
    return 0


def list_datasets(_: argparse.Namespace) -> int:
    for dataset in AVAILABLE_DATASETS:
        print(dataset)
    return 0


def _normalize_install_targets(targets: list[str]) -> list[str]:
    normalized: list[str] = []
    for target in targets:
        key = INSTALL_ALIASES.get(target, target)
        if key == "all":
            for name in INSTALL_REQUIREMENTS:
                if name not in normalized:
                    normalized.append(name)
            continue
        if key not in INSTALL_REQUIREMENTS:
            choices = ", ".join([*INSTALL_REQUIREMENTS, "all"])
            raise ValueError(f"Unknown install target '{target}'. Choose from: {choices}.")
        if key not in normalized:
            normalized.append(key)
    return normalized


def _requirements_for_targets(targets: list[str]) -> list[str]:
    requirements: list[str] = []
    for target in targets:
        for requirement in INSTALL_REQUIREMENTS[target]:
            if requirement not in requirements:
                requirements.append(requirement)
    return requirements


def install_backends(args: argparse.Namespace) -> int:
    targets = _normalize_install_targets(args.targets)
    requirements = _requirements_for_targets(targets)

    if args.installer == "uv":
        if shutil.which("uv") is None:
            raise RuntimeError("uv is not available on PATH. Use --installer pip instead.")
        command = ["uv", "pip", "install", "--python", sys.executable]
    else:
        command = [sys.executable, "-m", "pip", "install"]

    if args.upgrade:
        command.append("--upgrade")

    command.extend(requirements)
    print("Installing:", ", ".join(targets))
    print("Command:", " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        return 0

    subprocess.run(command, check=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="bm25-benchmark",
        description="Run BM25 benchmark evals from one CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command")

    models_parser = subparsers.add_parser("models", help="List available backends.")
    models_parser.set_defaults(func=list_models)

    datasets_parser = subparsers.add_parser("datasets", help="List known BEIR datasets.")
    datasets_parser.set_defaults(func=list_datasets)

    install_parser = subparsers.add_parser(
        "install",
        help="Install backend dependencies into the current CLI environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    install_parser.add_argument(
        "targets",
        nargs="+",
        choices=[*INSTALL_REQUIREMENTS, *INSTALL_ALIASES, "all"],
        help="Backend dependency set to install.",
    )
    install_parser.add_argument(
        "--installer",
        choices=["pip", "uv"],
        default="pip",
        help="Installer to use for the current Python interpreter.",
    )
    install_parser.add_argument(
        "--upgrade",
        action="store_true",
        help="Pass --upgrade to the installer.",
    )
    install_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the install command without running it.",
    )
    install_parser.set_defaults(func=install_backends)

    common = _common_eval_parser()
    bm25_params = _bm25_params_parser()
    eval_parser = subparsers.add_parser(
        "eval",
        aliases=["run"],
        help="Run an eval for one backend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    backend_parsers = eval_parser.add_subparsers(dest="backend", required=True)

    bm25s_parser = backend_parsers.add_parser(
        "bm25s",
        parents=[common],
        help="Evaluate bm25s.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bm25s_parser.add_argument(
        "--method",
        default="lucene",
        choices=["lucene", "atire", "robertson", "bm25l", "bm25+"],
        help="BM25S scoring method.",
    )
    bm25s_parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter.")
    bm25s_parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter.")
    bm25s_parser.add_argument("--delta", type=float, default=0.5, help="BM25 delta parameter.")
    bm25s_parser.add_argument(
        "--stopwords",
        default="en",
        choices=["en", "none"],
        help="Stopword handling.",
    )
    bm25s_parser.add_argument(
        "--stemmer-name",
        "--stemmer_name",
        dest="stemmer_name",
        default="snowball",
        choices=["snowball", "none"],
        help="Stemmer to use.",
    )
    bm25s_parser.add_argument(
        "--scorers",
        nargs="+",
        default=["jit"],
        choices=["uncompiled", "legacy", "jit"],
        help="Scorers to benchmark.",
    )
    bm25s_parser.add_argument(
        "--backends",
        nargs="+",
        default=["numba"],
        choices=["jax", "numba", "numpy"],
        help="Retrieval backends to benchmark.",
    )
    bm25s_parser.add_argument(
        "--use-function-tokenization",
        "--use_function_tokenization",
        dest="use_function_tokenization",
        action="store_true",
        help="Also benchmark bm25s.tokenize function tokenization.",
    )
    bm25s_parser.set_defaults(func=run_eval, backend="bm25s")

    rank_parser = backend_parsers.add_parser(
        "rank-bm25",
        aliases=["rank", "rank_bm25"],
        parents=[common],
        help="Evaluate rank-bm25.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    rank_parser.add_argument(
        "--method",
        default="rank",
        choices=["rank", "bm25l", "bm25+"],
        help="rank-bm25 variant.",
    )
    rank_parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="Number of sampled queries. Use 0 for all queries.",
    )
    rank_parser.add_argument("--verbose", action="store_true", help="Show progress bars.")
    rank_parser.set_defaults(func=run_eval, backend="rank-bm25")

    bm25_pt_parser = backend_parsers.add_parser(
        "bm25-pt",
        aliases=["bm25_pt"],
        parents=[common],
        help="Evaluate bm25-pt.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    bm25_pt_parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
        help="Query batch size for scoring.",
    )
    bm25_pt_parser.add_argument("--verbose", action="store_true", help="Show progress bars.")
    bm25_pt_parser.set_defaults(func=run_eval, backend="bm25-pt")

    pyserini_parser = backend_parsers.add_parser(
        "pyserini",
        parents=[common, bm25_params],
        help="Evaluate Pyserini.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pyserini_parser.set_defaults(func=run_eval, backend="pyserini")

    elastic_parser = backend_parsers.add_parser(
        "elastic",
        aliases=["elastic-bm25"],
        parents=[common, bm25_params],
        help="Evaluate Elasticsearch BM25.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    elastic_parser.add_argument(
        "--hostname",
        default="localhost",
        help="Elasticsearch hostname.",
    )
    elastic_parser.set_defaults(func=run_eval, backend="elastic")

    pisa_parser = backend_parsers.add_parser(
        "pisa",
        parents=[common, bm25_params],
        help="Evaluate PISA.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    pisa_parser.set_defaults(func=run_eval, backend="pisa")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        return 0

    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
