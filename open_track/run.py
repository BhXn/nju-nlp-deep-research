from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_SUBMISSIONS = [
    ("v6", "runs/deep_research_submission_v6_docref.jsonl"),
    ("v8", "runs/deep_research_submission_v8_repeatcap.jsonl"),
    ("old_v11", "runs/deep_research_submission_v11_overnight.jsonl"),
    ("old_v12", "runs/deep_research_submission_v12_overnight_no_react_verify.jsonl"),
    ("old_v13", "runs/deep_research_submission_v13_overnight_heuristic.jsonl"),
    ("v14", "runs/deep_research_submission_v14_balanced_overnight.jsonl"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the final OpenTrack relaxed fusion pipeline.")
    parser.add_argument("--dataset", default="browsecomp_plus_hard50.jsonl", help="Dataset JSONL path.")
    parser.add_argument("--model", default="qwen_auto", help="OpenAI-compatible served model name.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="dummy", help="API key for the local endpoint.")
    parser.add_argument(
        "--output",
        default="runs/deep_research_submission_fused_v6_relaxed.jsonl",
        help="Fused OpenTrack submission output path.",
    )
    parser.add_argument(
        "--eval-output",
        default="runs/deep_research_eval_fused_v6_relaxed.jsonl",
        help="Evaluation output path.",
    )
    parser.add_argument(
        "--analysis-output",
        default="runs/deep_research_analysis_fused_v6_relaxed.json",
        help="Analysis output path.",
    )
    parser.add_argument(
        "--submission",
        action="append",
        default=[],
        help="Extra or replacement source submission in label=path format. May be repeated.",
    )
    parser.add_argument("--base-label", default="v6", help="Base run label for conservative fusion.")
    parser.add_argument("--override-confidence", type=int, default=80, help="Judge confidence gate.")
    parser.add_argument("--override-score-margin", type=int, default=4, help="Judge score-margin gate.")
    parser.add_argument("--skip-missing", action="store_true", help="Skip missing default source submissions.")
    parser.add_argument("--no-eval", action="store_true", help="Only write fused submission; skip eval/analyze.")
    return parser.parse_args()


def source_specs(args: argparse.Namespace) -> list[str]:
    specs = [f"{label}={path}" for label, path in DEFAULT_SUBMISSIONS]
    if args.submission:
        specs.extend(args.submission)

    existing_specs: list[str] = []
    missing: list[str] = []
    source_run_dir = Path(__file__).resolve().parent / "source_runs"
    for spec in specs:
        label, path_text = spec.split("=", 1) if "=" in spec else (Path(spec).stem, spec)
        path = Path(path_text)
        if path.exists():
            existing_specs.append(f"{label}={path_text}")
            continue

        fallback = source_run_dir / path.name
        if fallback.exists():
            existing_specs.append(f"{label}={fallback}")
        else:
            missing.append(f"{label}={path_text}")

    if missing and not args.skip_missing:
        joined = "\n  ".join(missing)
        raise FileNotFoundError(
            "Missing source submissions. Re-run the corresponding agent runs, "
            "or pass --skip-missing for an ablation-only fusion:\n  " + joined
        )
    return existing_specs


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    module_paths = []
    if (script_dir / "agent").exists():
        module_paths.append(str(script_dir))
    if (repo_root / "agent").exists():
        module_paths.append(str(repo_root))

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        module_paths.append(existing_pythonpath)
    if module_paths:
        env["PYTHONPATH"] = os.pathsep.join(module_paths)
    subprocess.run(command, check=True, env=env)


def main() -> None:
    args = parse_args()
    submissions = source_specs(args)
    if not any(spec.startswith(f"{args.base_label}=") for spec in submissions):
        raise ValueError(f"Base label {args.base_label!r} is not present in available submissions: {submissions}")

    fuse_cmd = [
        sys.executable,
        "-m",
        "agent.fuse_deep_research_runs",
        "--dataset",
        args.dataset,
        "--base-label",
        args.base_label,
        "--override-confidence",
        str(args.override_confidence),
        "--override-score-margin",
        str(args.override_score_margin),
        "--protected-base-extra-confidence",
        "0",
        "--protected-base-extra-margin",
        "0",
        "--model",
        args.model,
        "--base-url",
        args.base_url,
        "--api-key",
        args.api_key,
        "--output",
        args.output,
    ]
    for spec in submissions:
        fuse_cmd.extend(["--submission", spec])
    run_command(fuse_cmd)

    if args.no_eval:
        return

    run_command(
        [
            sys.executable,
            "-m",
            "agent.eval",
            "--submission",
            args.output,
            "--dataset",
            args.dataset,
            "--model",
            args.model,
            "--base-url",
            args.base_url,
            "--api-key",
            args.api_key,
            "--output",
            args.eval_output,
        ]
    )
    run_command(
        [
            sys.executable,
            "-m",
            "agent.analyze_deep_research_run",
            "--submission",
            args.output,
            "--eval",
            args.eval_output,
            "--dataset",
            args.dataset,
            "--output",
            args.analysis_output,
        ]
    )


if __name__ == "__main__":
    main()
