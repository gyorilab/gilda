"""This is a script that helps deal with diffs of benchmarks."""

from pathlib import Path
from textwrap import dedent

import click
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

HERE = Path(__file__).parent.resolve()
RESULTS = HERE.joinpath('results', 'bioid_performance')


@click.command()
@click.option("--reference-version", default="0.8.3")
@click.option("--comparison-version", default="0.8.4")
@click.option("--key", default="exists_correct")
def main(reference_version: str, comparison_version: str, key: str):
    reference_json = RESULTS.joinpath(reference_version, "benchmark.json")
    df1 = pd.read_json(reference_json, orient="record")
    comparison_json = RESULTS.joinpath(comparison_version, "benchmark.json")
    df2 = pd.read_json(comparison_json, orient="record")
    output = RESULTS.joinpath(f"{reference_version}_{comparison_version}", key)
    output.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    venn2(
        [
            set(df1[df1[key]].index),
            set(df2[df2[key]].index),
        ],
        [
            reference_version,
            comparison_version,
        ],
        ax=ax,
    )
    fig.savefig(output.joinpath("comparison.svg"))

    # Here are the texts that grounded in the new version but not the old
    new_idx = df2[key] & ~df1[key]
    newly_grounded = df2[new_idx].text.value_counts()
    newly_grounded.to_csv(output.joinpath("improvements.tsv"), sep='\t')
    new_unique_count = df2[new_idx].text.nunique()

    # Here are the texts that grounded in the old version but not the new
    regressions_idx = df1[key] & ~df2[key]
    regressions = df2[regressions_idx].text.value_counts()
    regressions.to_csv(output.joinpath("regressions.tsv"), sep='\t')
    regression_unique_count = df2[regressions_idx].text.nunique()

    # Here are the texts that neither versions groudned right
    misses_idx = ~df1[key] & ~df2[key]
    misses = df2[misses_idx].text.value_counts()
    misses.to_csv(output.joinpath("misses.tsv"), sep='\t')
    misses_unique_count = df2[misses_idx].text.nunique()

    hits_idx = df1[key] & df2[key]
    hits_unique_count = df2[hits_idx].text.nunique()

    print(dedent(f"""\
    Analysis of "{key}":
    
    {new_idx.sum():,} success rows ({new_unique_count:,} unique) in v{comparison_version} but not v{reference_version}
    {regressions_idx.sum():,} success rows ({regression_unique_count:,} unique) in v{reference_version} but not v{comparison_version}
    {misses.sum():,} rows missed ({misses_unique_count:,} unique) in both
    {hits_idx.sum():,} successes rows ({hits_unique_count:,} unique) in both
    """))


if __name__ == '__main__':
    main()
