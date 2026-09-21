"""Command line: `quotesim report` regenerates the README tables, `quotesim run --seed N` prints one
attribution waterfall. Both are synthetic and seeded; nothing reads market data."""

from __future__ import annotations

import argparse
import sys
import time

from quotesim import __version__, report
from quotesim.adverse import toxicity
from quotesim.flow import FlowParams
from quotesim.hedger import BandHedger
from quotesim.sim import SimConfig, run


def _run_cmd(a: argparse.Namespace) -> int:
    fair = report.strip()
    params = FlowParams(informed_frac=a.informed, **report.FLOW)
    ps = report.price_quoter(a.gamma, a.seconds)
    quoter = ps if a.space == "price" else report.vol_quoter(a.gamma, ps, fair)
    r = run(SimConfig(seconds=a.seconds, seed=a.seed, h_markout=a.h), quoter, BandHedger(a.band), fair, params)
    print(f"quotesim {__version__} run: seed {a.seed}, {a.seconds:g} s, {a.space}-space gamma {a.gamma:g}, informed {a.informed:g}, "
          f"band ${a.band:g}, {r.n_fills} fills, {r.n_hedges} hedges, {r.n_candidates} candidates")
    print(r.attribution.waterfall())
    print()
    print("toxicity at the canonical horizon ($ per contract):")
    print(toxicity(r).to_string(float_format=lambda x: f"{x:+.4f}"))
    return 0


def _report_cmd(a: argparse.Namespace) -> int:
    sizes = report.ReportSizes.small() if a.small else report.ReportSizes()
    t0 = time.perf_counter()
    if a.check:
        bad = report.check_readme(a.readme, sizes)
        for name, items in bad.items():
            print(f"{a.readme} section '{name}' differs from a fresh `quotesim report`:", file=sys.stderr)
            for it in items:
                print(f"  {it}", file=sys.stderr)
        print(f"quotesim report --check: {time.perf_counter() - t0:.1f} s wall", file=sys.stderr)
        if bad:
            return 1
        print(f"{a.readme}: every section reproduces (gap ceilings within one decade)", file=sys.stderr)
        return 0
    if a.stdout:
        blocks = report.render(report.build(sizes))
        for name, body in blocks.items():
            print(report.BEGIN.format(name=name))
            print(body)
            print(report.END.format(name=name))
            print()
    else:
        blocks = report.update_readme(a.readme, sizes)
        print(f"updated {len(blocks)} sections in {a.readme}", file=sys.stderr)
    print(f"quotesim report: {time.perf_counter() - t0:.1f} s wall", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="quotesim", description="options quoting simulator with synthetic flow")
    p.add_argument("--version", action="version", version=f"quotesim {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="one seeded run and its attribution waterfall")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--seconds", type=float, default=3600.0)
    r.add_argument("--space", choices=("price", "vol"), default="price")
    r.add_argument("--gamma", type=float, default=10.0)
    r.add_argument("--informed", type=float, default=0.1)
    r.add_argument("--band", type=float, default=10.0)
    r.add_argument("--h", type=float, default=60.0, help="markout horizon in seconds")
    r.set_defaults(fn=_run_cmd)
    rp = sub.add_parser("report", help="regenerate the README tables between the quotesim markers")
    rp.add_argument("--readme", default="README.md")
    rp.add_argument("--stdout", action="store_true", help="print the blocks instead of editing the README")
    rp.add_argument("--small", action="store_true", help="the tests' sizes (seconds, not minutes)")
    rp.add_argument("--check", action="store_true",
                    help="regenerate and compare instead of writing: exit 1 unless every token matches, gap ceilings within one decade")
    rp.set_defaults(fn=_report_cmd)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
