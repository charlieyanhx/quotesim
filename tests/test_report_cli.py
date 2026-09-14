"""report.py and cli.py: the A-S 2008 Tables 1-3 reproduction (their parameters, 1,000 sims), the five
sections at the tests' sizes, markdown rendering, marker replacement, byte-identical regeneration, the CLI,
and the README token rule. Oracles: docs/PLAN.md 'A-S 2008 Tables 1-3' (inventory strategy: P&L sd 6.6 vs
12.7 and |q| sd 2.9 vs 8.4 at gamma = 0.1; mean 31.4 vs 44.0 at gamma = 1; direction asserted, magnitudes
within ~15 %) and research report_a3e5dd section 1 (A-S Table 1 spreads 1.49 / 1.35 / 3.02)."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quotesim import cli, report
from quotesim.report import (
    AS_PAPER,
    SECTIONS,
    ReportSizes,
    as_benchmark,
    md_table,
    pow10_ceiling,
    replace_sections,
    update_readme,
)

REPO = Path(__file__).resolve().parents[1]
SMALL = ReportSizes.small()
BANNED = re.compile(r"profit|\bedge\b|sharpe|arbitrage-free|validated", re.IGNORECASE)


@pytest.fixture(scope="module")
def built():
    return report.build(SMALL)


# ----------------------------------------------------------------------------------------------------------
# A-S 2008 Tables 1-3
# ----------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("gamma", [0.01, 0.1, 1.0])
def test_as2008_tables_direction_and_magnitude(gamma):
    """A-S 2008 Tables 1-3 with their parameters and 1,000 paths (seed 0): the inventory strategy has the
    lower P&L sd and the lower final-|q| sd at every gamma, its mean is lower at gamma = 1 (31.4 vs 44.0),
    and every sd / mean is within 15 % of the paper's; the average spread matches Table 1 to 1 %."""
    res = as_benchmark(gamma, n_sims=1000, seed=0)
    inv, sym = res["inventory"], res["symmetric"]
    assert inv[2] < sym[2] and inv[4] < sym[4]
    for strategy, ours in res.items():
        paper = AS_PAPER[gamma][strategy]
        assert ours[0] == pytest.approx(paper[0], rel=0.01)
        assert ours[1] == pytest.approx(paper[1], rel=0.15)
        assert ours[2] == pytest.approx(paper[2], rel=0.15)
        assert ours[4] == pytest.approx(paper[4], rel=0.15)
    if gamma == 1.0:
        assert inv[1] < sym[1]


def test_as2008_is_seeded_and_symmetric_uses_the_average_spread():
    a, b = as_benchmark(0.1, n_sims=100, seed=4), as_benchmark(0.1, n_sims=100, seed=4)
    assert a == b
    assert a["inventory"][0] == a["symmetric"][0]
    assert as_benchmark(0.1, n_sims=100, seed=5) != a


# ----------------------------------------------------------------------------------------------------------
# sections at the tests' sizes
# ----------------------------------------------------------------------------------------------------------


def test_identity_section_every_run_inside_the_bar(built):
    df = built["identity"]
    assert list(df["quoter"]) == ["PriceSpace glft_asym", "VolSpace matched"]
    half = SMALL.identity_seeds // 2
    assert list(df["runs with |gap| < 1e-9"]) == [f"{half}/{half}"] * 2
    for col in ("max |gap|", "max fill-split gap", "max theta-split gap"):
        for cell in df[col]:
            assert cell == "0" or int(cell.split("e")[1]) <= -9, (col, cell)
    assert (df["fills per run (mean)"] > 0).all()


def test_sweep_section_shape_and_paired_columns(built):
    df = built["sweep"]
    assert len(df) == len(report.GAMMAS) * len(report.INFORMED) * len(report.BANDS)
    assert set(df["gamma"]) == set(report.GAMMAS) and set(df["informed"]) == set(report.INFORMED)
    for cell in df["sign"]:
        pos, nz = (int(x) for x in cell.split("/"))
        assert 0 <= pos <= nz <= SMALL.sweep_seeds
    for col in ("realised: median", "spread: median", "n_fills: median"):
        assert np.isfinite(df[col]).all()
    assert (df["hedge_cost: median"].abs() < 1e3).all()


def test_markout_section_rows_and_toxicity_shares(built):
    mk, tox = built["markouts"], built["toxicity"]
    assert set(mk["horizon_s"]) == {1.0, 10.0, 60.0, 300.0} and len(mk) == 12
    assert list(tox["class"]) == ["all", "uninformed", "informed"]
    assert tox.loc[tox["class"] == "all", "share"].item() == 1.0
    assert tox.loc[tox["class"] != "all", "n"].sum() == tox.loc[tox["class"] == "all", "n"].item()


def test_ww_section_frontier_is_monotone_and_band_scales_as_gamma_to_minus_third(built):
    fr, ww = built["frontier"], built["ww"]
    assert list(fr["band $ delta"]) == list(report.WW_BANDS)
    assert (np.diff(fr["hedges per run"]) <= 0).all()
    assert (np.diff(fr["RMS residual $ delta"]) >= 0).all()
    assert fr["RMS residual $ delta"].iloc[0] == 0.0
    h = ww.set_index("gamma")["WW half-band, shares per lot"]
    assert h[1.0] / h[10.0] == pytest.approx(10 ** (1 / 3), rel=1e-9)
    assert h[10.0] / h[50.0] == pytest.approx(5 ** (1 / 3), rel=1e-9)


def test_matched_half_spreads_sum_equal_but_shape_differs():
    """match_spreads equates sum Vega hs_vol with sum hs_$; the vol quotes are Black at sigma -+ hs_vol, so the
    $ sums agree to first order in hs_vol while the shapes differ (flat in $ vs flat in vol)."""
    hs_p, hs_v = report.matched_half_spreads(10.0, 60.0)
    assert np.sum(hs_v) == pytest.approx(np.sum(hs_p), rel=0.05)
    assert hs_p.max() - hs_p.min() < 0.01 and hs_v.max() > 2 * hs_v.min()


# ----------------------------------------------------------------------------------------------------------
# rendering
# ----------------------------------------------------------------------------------------------------------


def test_pow10_ceiling():
    assert pow10_ceiling(3.6e-13) == "< 1e-12"
    assert pow10_ceiling(-3.6e-13) == "< 1e-12"
    assert pow10_ceiling(1e-9) == "< 1e-8"
    assert pow10_ceiling(0.0) == "0"
    assert pow10_ceiling(0.5) == "< 1e0"


def test_md_table_alignment_ints_nans_and_formats():
    df = pd.DataFrame({"name": ["a", "b"], "n": [1, 20], "x": [0.12345, float("nan")]})
    out = md_table(df, fmt={"x": "{:.2f}"}).splitlines()
    assert out[0] == "| name | n | x |"
    assert out[1] == "|---|---:|---:|"
    assert out[2] == "| a | 1 | 0.12 |" and out[3] == "| b | 20 |  |"
    idx = md_table(df.set_index("name"), index=True).splitlines()
    assert idx[0] == "| name | n | x |" and idx[1] == "|---|---:|---:|" and idx[2].startswith("| a | 1 |")


def test_replace_sections_keeps_markers_and_outside_text_and_raises_on_missing():
    text = "head\n<!-- quotesim:begin:x -->\nold\n<!-- quotesim:end:x -->\ntail\n"
    new = replace_sections(text, {"x": "new body"})
    assert new == "head\n<!-- quotesim:begin:x -->\nnew body\n<!-- quotesim:end:x -->\ntail\n"
    assert replace_sections(new, {"x": "new body"}) == new
    with pytest.raises(ValueError, match="no markers"):
        replace_sections(text, {"y": "z"})


def test_render_has_every_section_and_no_banned_token(built):
    blocks = report.render(built)
    assert set(blocks) == set(SECTIONS)
    for name, body in blocks.items():
        assert body.startswith("|"), name
        assert not BANNED.search(body), (name, BANNED.search(body).group(0))


def test_update_readme_twice_is_byte_identical(tmp_path):
    p = tmp_path / "README.md"
    skeleton = "# t\n\n" + "\n\n".join(f"{report.BEGIN.format(name=n)}\n{report.END.format(name=n)}" for n in SECTIONS) + "\n\nend\n"
    p.write_text(skeleton, encoding="utf-8")
    update_readme(p, SMALL)
    first = p.read_bytes()
    update_readme(p, SMALL)
    assert p.read_bytes() == first
    text = first.decode()
    assert text.startswith("# t\n\n") and text.endswith("\n\nend\n")
    assert text.count("<!-- quotesim:begin:") == len(SECTIONS) and "| gamma |" in text


# ----------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------


def test_cli_run_prints_the_waterfall(capsys):
    assert cli.main(["run", "--seed", "1", "--seconds", "30", "--space", "vol", "--informed", "0.3"]) == 0
    out = capsys.readouterr().out
    assert "seed 1, 30 s, vol-space" in out and "= realised" in out and "RESID" in out and "toxicity" in out


def test_cli_report_stdout_and_readme(tmp_path, capsys):
    assert cli.main(["report", "--small", "--stdout"]) == 0
    out = capsys.readouterr().out
    assert all(report.BEGIN.format(name=n) in out and report.END.format(name=n) in out for n in SECTIONS)
    p = tmp_path / "R.md"
    p.write_text("\n".join(f"{report.BEGIN.format(name=n)}\n{report.END.format(name=n)}" for n in SECTIONS), encoding="utf-8")
    assert cli.main(["report", "--small", "--readme", str(p)]) == 0
    assert "| gamma |" in p.read_text(encoding="utf-8")
    with pytest.raises(SystemExit):
        cli.main(["--version"])


# ----------------------------------------------------------------------------------------------------------
# the committed README
# ----------------------------------------------------------------------------------------------------------


def test_readme_has_the_markers_and_none_of_the_banned_tokens():
    text = (REPO / "README.md").read_text(encoding="utf-8")
    for n in SECTIONS:
        assert report.BEGIN.format(name=n) in text and report.END.format(name=n) in text
    hits = sorted({m.group(0).lower() for m in BANNED.finditer(text)})
    assert hits == [], hits
    assert "synthetic" in text and "no queue" in text and "MIT © Hanxiong (Charlie) Yan" in text
    assert f"{ReportSizes().identity_seeds} seeds" in text
