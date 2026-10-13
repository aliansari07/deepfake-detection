"""Render every thesis table from JSON (PLAN task D3).

No number in the thesis should be typed by hand. Hand-copied numbers drift: this project has
already had a 0.9084 that was 0.9047 somewhere else, and an A4 AUC quoted as 0.9653 from a training
log when the evaluation of record said 0.9665. Every table below is generated from the JSON written
by the evaluation tools, so regenerating is the only way to change a number, and a stale table is
visible as a missing input rather than a plausible-looking wrong figure.

Inputs (all optional — a missing one produces a clearly-marked gap, never a silent omission):

    results/comparison_test.json      FF++ test, from tools/compare_runs.py
    results/comparison_celebdf.json   Celeb-DF official, same tool
    results/comparison_night1.json    FF++ validation
    results/freq_probe.json           R1 frequency-domain linear probes
    results/crf_curves.json           C1 video-level CRF robustness
    results/TEST_ACCESS_LOG.md        test-split access record

Usage:
    python tools/make_tables.py                  # writes results/TABLES.md
    python tools/make_tables.py --check          # non-zero exit if regenerating would change it
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS = "results"
OUT = os.path.join(RESULTS, "TABLES.md")

#: Display names, so the tables do not carry directory names like "eval_A2-20.best_test".
PRETTY = {
    "a220": "A2 (spatial only)",
    "m120": "M1 (full model)",
    "a120": "A1 (no frequency)",
    "a420": "A4 (no augmentation)",
    "a320": "A3 (log-DCT)",
    # Celeb-DF runs are tagged differently, and "a2" must not match "a420" first.
    "a2": "A2 (spatial only)",
}


def load(name):
    """Load a JSON file from ``results/``, or return ``None`` if it is missing."""
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def short(key):
    """'eval_A2-20.best_test' or 'eval_celebdf_official_a2' -> a readable model name.

    Longest key first: "a2" is a substring of nothing here, but "a420" must not be matched by a
    shorter pattern, and dict order alone would not guarantee that.
    """
    flat = key.lower().replace("-", "").replace("_", "").replace(".", "")
    for raw in sorted(PRETTY, key=len, reverse=True):
        if raw in flat:
            return PRETTY[raw]
    return key


def missing(title, filename, how):
    """Placeholder section for a results file that has not been generated."""
    return [
        f"### {title}",
        "",
        f"> **Not generated.** `{RESULTS}/{filename}` does not exist yet.",
        f"> Produce it with: `{how}`",
        "",
    ]


def ci(entry, places=4):
    """Format a stored confidence interval as ``point [low, high]``."""
    return (
        f"{entry['point']:.{places}f} [{entry['ci_low']:.{places}f}, {entry['ci_high']:.{places}f}]"
    )


def table_models(data, title, note=""):
    """Markdown table of per-model metrics with confidence intervals."""
    out = [f"### {title}", ""]
    if note:
        out += [note, ""]
    names = list(data["models"])
    out.append("| Model | AUC (95% CI) | Accuracy | Balanced accuracy | Threshold |")
    out.append("|---|---|---|---|---|")
    for n in names:
        m = data["models"][n]
        out.append(
            f"| {short(n)} | {ci(m['auc'])} | {m['accuracy']['point'] * 100:.2f}% | "
            f"{m['balanced_accuracy']['point'] * 100:.2f}% | {m['threshold']:.4f} |"
        )
    out.append("")
    if data.get("comparisons"):
        ref = short(data["comparisons"][0]["name_b"])
        out += [
            f"Paired AUC differences against **{ref}**, "
            f"{data['comparisons'][0]['resamples']} bootstrap resamples:",
            "",
            "| Comparison | Difference | 95% CI | Paired rule | Marginal rule |",
            "|---|---|---|---|---|",
        ]
        for c in data["comparisons"]:
            out.append(
                f"| {short(c['name_a'])} − {short(c['name_b'])} | {c['difference']:+.4f} | "
                f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                f"{'excludes zero' if c['paired_excludes_zero'] else '**spans zero**'} | "
                f"{'disjoint' if c['marginal_disjoint'] else '**overlap**'} |"
            )
        out.append("")
        if any(not c["rules_agree"] for c in data["comparisons"]):
            out += [
                "> The two claim rules disagree on at least one row above. Decision **D6** "
                "(`audit-fyp2.md` §14.4) decides which governs.",
                "",
            ]
    return out


def table_probe(data):
    """Markdown table of the frequency-domain linear probe results."""
    out = [
        "### R1 — frequency-domain linear probes (validation AUC)",
        "",
        "A spatial CNN reaches 0.91–0.94 on the same videos. These are floors, not ceilings: a "
        "linear probe says what is *trivially* extractable.",
        "",
    ]
    methods = sorted({m for s in data["sets"].values() for m in s["per_method"]})
    out.append("| Feature set | Features | Overall | " + " | ".join(methods) + " |")
    out.append("|---|---|---|" + "---|" * len(methods))
    for name, s in data["sets"].items():
        cells = " | ".join(f"{s['per_method'].get(m, float('nan')):.4f}" for m in methods)
        out.append(f"| {name} | {s['n_features']} | **{s['overall_auc']:.4f}** | {cells} |")
    out.append("")
    return out


def table_crf(data):
    """Markdown table of the video-level compression (CRF) results."""
    out = [
        "### C1 — video-level compression robustness",
        "",
        f"H.264 re-encode at each CRF, faces re-detected from the re-encoded file, "
        f"{data['n']} videos from the **{data['split']}** split "
        f"({data['n_real']} real). The claim rests on the *slope*, not the absolute level.",
        "",
    ]
    levels = [str(c) for c in data["crf_levels"]]
    out.append("| Model | " + " | ".join(f"CRF {c}" for c in levels) + " | Drop |")
    out.append("|---|" + "---|" * (len(levels) + 1))
    for name, m in data["models"].items():
        cells = []
        for c in levels:
            e = m["curve"].get(c)
            cells.append(f"{e['auc']:.4f}" if e else "—")
        drop = m.get("degradation", {}).get("auc_drop")
        out.append(
            f"| {short(name)} | "
            + " | ".join(cells)
            + " | "
            + (f"**{drop:+.4f}**" if drop is not None else "—")
            + " |"
        )
    out.append("")
    no_face = {
        n: sum(e.get("n_no_face", 0) for e in m["curve"].values())
        for n, m in data["models"].items()
    }
    if any(no_face.values()):
        out += [
            "Videos where no face survived the compression are excluded from that level's "
            "AUC and counted here: "
            + ", ".join(f"{short(n)} {v}" for n, v in no_face.items() if v)
            + ".",
            "",
        ]
    return out


def access_log_summary():
    """Summarise the test-split access log as Markdown."""
    path = os.path.join(RESULTS, "TEST_ACCESS_LOG.md")
    if not os.path.exists(path):
        return ["### Test-split access", "", "> No test-split read has been recorded.", ""]
    from src.test_access import _rows  # one parser for this format, not two

    rows = _rows(path)
    repeats = [r for r in rows if r["kind"] == "repeat"]
    return [
        "### Test-split access",
        "",
        f"The FF++ test split has been read **{len(rows)}** time(s), "
        f"{len(repeats)} of them repeats. One read per model is the rule "
        f"(`{RESULTS}/TEST_ACCESS_LOG.md` holds the full record with checkpoint hashes).",
        "",
    ]


def build():
    """Assemble the full ``results/TABLES.md`` document."""
    lines = [
        "# Generated results tables",
        "",
        "**Every number here is generated.** Do not edit this file; edit the evaluation that "
        "produced the JSON and run `python tools/make_tables.py` again (PLAN task D3).",
        "",
        "Regenerate with:",
        "",
        "```powershell",
        "python tools/make_tables.py",
        "```",
        "",
        "---",
        "",
    ]

    test = load("comparison_test.json")
    lines += (
        table_models(
            test,
            "FF++ test set — FINAL",
            "One read per model, each at its own validation-calibrated threshold. "
            "**These numbers are final; nothing may be tuned against them.**",
        )
        if test
        else missing(
            "FF++ test set — FINAL",
            "comparison_test.json",
            "python evaluate.py --checkpoint <ckpt> --split test --final, then "
            "tools/compare_runs.py --runs checkpoints/eval_*_test",
        )
    )

    lines += access_log_summary()

    celeb = load("comparison_celebdf.json")
    lines += (
        table_models(
            celeb,
            "Celeb-DF v2 — official 518-video protocol",
            "Cross-dataset generalisation (criterion S2, target >75% accuracy). "
            "Unseen by every model, so FYP-1 and FYP-2 may be compared here.",
        )
        if celeb
        else missing(
            "Celeb-DF v2 — official protocol",
            "comparison_celebdf.json",
            "python cross_eval.py --checkpoint <ckpt> --protocol official --tag <tag>",
        )
    )

    val = load("comparison_night1.json")
    lines += (
        table_models(
            val,
            "FF++ validation — model selection",
            "Used for choosing epochs and thresholds. Not a headline result.",
        )
        if val
        else missing(
            "FF++ validation",
            "comparison_night1.json",
            "python tools/compare_runs.py --checkpoints checkpoints/*.best.pt",
        )
    )

    crf = load("crf_curves.json")
    lines += (
        table_crf(crf)
        if crf
        else missing(
            "C1 — video-level compression robustness",
            "crf_curves.json",
            "python compress_video_test.py --limit 140",
        )
    )

    probe = load("freq_probe.json")
    lines += (
        table_probe(probe)
        if probe
        else missing(
            "R1 — frequency-domain probes", "freq_probe.json", "python tools/freq_probe.py"
        )
    )

    return "\n".join(lines).rstrip() + "\n"


def main():
    """Write ``results/TABLES.md``, or check that it is current."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the file on disk is not what would be generated",
    )
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    text = build()
    if args.check:
        current = open(args.out, encoding="utf-8").read() if os.path.exists(args.out) else ""
        if current != text:
            print(f"{args.out} is out of date — run: python tools/make_tables.py")
            return 1
        print(f"{args.out} is up to date")
        return 0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write(text)
    gaps = text.count("**Not generated.**")
    print(
        f"written: {args.out}"
        + (f"   ({gaps} table(s) still missing their input)" if gaps else "   (all tables present)")
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
