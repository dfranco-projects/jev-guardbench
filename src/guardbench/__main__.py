import argparse
import asyncio
from datetime import date
from pathlib import Path

from guardbench.config import build_guard, freeze_or_verify, load_config, load_rows
from guardbench.report import load_all, render
from guardbench.runner import results_path, run_guard


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    names = args.guards.split(",") if args.guards else cfg.run
    rows = load_rows(cfg)
    print(f"{len(rows)} rows from {', '.join(cfg.sources)}")
    if cfg.manifest:
        freeze_or_verify(cfg.manifest, rows)
    for name in names:
        guard = build_guard(name, cfg.guards)
        for conc in cfg.concurrency:
            out = results_path(cfg.results_dir, name, conc)
            ran, failed = asyncio.run(run_guard(guard, rows, out, conc))
            print(f"{name} c={conc}: ran {ran}, failed {failed} -> {out}")


def cmd_freeze(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if not cfg.manifest:
        raise SystemExit(f"{args.config} has no `manifest` path")
    rows = load_rows(cfg)
    written = freeze_or_verify(cfg.manifest, rows)
    print(f"{len(rows)} rows {'written to' if written else 'match'} {cfg.manifest}")


def cmd_report(args: argparse.Namespace) -> None:
    text = render(load_all(args.results), baseline=args.baseline)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(prog="guardbench")
    sub = parser.add_subparsers(required=True)

    run = sub.add_parser("run", help="score rows with guards")
    run.add_argument("config", type=Path)
    run.add_argument("--guards", help="comma-separated guard names (default: config `run`)")
    run.set_defaults(func=cmd_run)

    freeze = sub.add_parser("freeze", help="write the config's row manifest, or verify it")
    freeze.add_argument("config", type=Path)
    freeze.set_defaults(func=cmd_freeze)

    report = sub.add_parser("report", help="render results as markdown")
    report.add_argument("--results", type=Path, default=Path("results"))
    report.add_argument("--baseline", help="guard to compare every other guard against")
    report.add_argument("--out", type=Path, default=Path("reports") / f"{date.today()}.md")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
