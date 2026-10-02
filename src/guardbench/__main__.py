import argparse
import asyncio
from collections import defaultdict
from datetime import date
from pathlib import Path

from guardbench.attacks import variants
from guardbench.config import build_guard, freeze_or_verify, load_config, load_rows
from guardbench.data import Row, sample, split_of
from guardbench.hypotheses import Roles, render_hypotheses
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


def cmd_repeat(args: argparse.Namespace) -> None:
    """Determinism check: score the same test rows several times with every non-cascade guard."""
    cfg = load_config(args.config)
    rows = load_rows(cfg)
    if cfg.manifest:
        freeze_or_verify(cfg.manifest, rows)
    picked = sample([r for r in rows if split_of(r.id) == "test"], args.rows, cfg.seed)
    names = [n for n in cfg.run if cfg.guards[n]["type"] != "cascade"]
    for k in range(1, args.times + 1):
        for name in names:
            out = results_path(cfg.results_dir / "repeat" / f"r{k}", name, 1)
            ran, failed = asyncio.run(run_guard(build_guard(name, cfg.guards), picked, out, 1))
            print(f"run {k} {name}: ran {ran}, failed {failed} -> {out}")


def cmd_attack(args: argparse.Namespace) -> None:
    """Attacks on the guard: violating test rows, unchanged and with each attack appended."""
    cfg = load_config(args.config)
    rows = load_rows(cfg)
    if cfg.manifest:
        freeze_or_verify(cfg.manifest, rows)
    picked = sample([r for r in rows if split_of(r.id) == "test" and r.label], args.rows, cfg.seed)
    by_variant: dict[str, list[Row]] = defaultdict(list)
    for row in picked:
        for name, variant in variants(row).items():
            by_variant[name].append(variant)
    names = [n for n in cfg.run if cfg.guards[n]["type"] != "cascade"]
    for variant, vrows in by_variant.items():
        for name in names:
            out = results_path(cfg.results_dir / "attack" / variant, name, 1)
            ran, failed = asyncio.run(run_guard(build_guard(name, cfg.guards), vrows, out, 1))
            print(f"{variant} {name}: ran {ran}, failed {failed} -> {out}")


def cmd_freeze(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    if not cfg.manifest:
        raise SystemExit(f"{args.config} has no `manifest` path")
    rows = load_rows(cfg)
    written = freeze_or_verify(cfg.manifest, rows)
    print(f"{len(rows)} rows {'written to' if written else 'match'} {cfg.manifest}")


def cmd_report(args: argparse.Namespace) -> None:
    recs = load_all(args.results)
    text = render(recs, baseline=args.baseline)
    if args.config:
        cfg = load_config(args.config)
        h3 = args.results / "h3.md"
        roles = Roles.from_guards(cfg.guards, cfg.run)
        h3_text = h3.read_text(encoding="utf-8") if h3.exists() else "Not run yet."
        repeats = [load_all(d) for d in sorted((args.results / "repeat").glob("r*"))]
        attack_dir = args.results / "attack"
        attacked = {d.name: load_all(d) for d in sorted(attack_dir.glob("*")) if d.is_dir()}
        text = render_hypotheses(recs, roles, h3_text, repeats, attacked) + "\n\n" + text
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

    repeat = sub.add_parser("repeat", help="score the same test rows several times")
    repeat.add_argument("config", type=Path)
    repeat.add_argument("--rows", type=int, default=200)
    repeat.add_argument("--times", type=int, default=3)
    repeat.set_defaults(func=cmd_repeat)

    attack = sub.add_parser("attack", help="score violating rows with attacks on the guard")
    attack.add_argument("config", type=Path)
    attack.add_argument("--rows", type=int, default=200)
    attack.set_defaults(func=cmd_attack)

    freeze = sub.add_parser("freeze", help="write the config's row manifest, or verify it")
    freeze.add_argument("config", type=Path)
    freeze.set_defaults(func=cmd_freeze)

    report = sub.add_parser("report", help="render results as markdown")
    report.add_argument("--results", type=Path, default=Path("results"))
    report.add_argument("--baseline", help="guard to compare every other guard against")
    report.add_argument("--config", type=Path, help="run config; adds the hypothesis verdicts")
    report.add_argument("--out", type=Path, default=Path("reports") / f"{date.today()}.md")
    report.set_defaults(func=cmd_report)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
