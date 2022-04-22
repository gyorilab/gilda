"""This is a script that helps deal with diffs of benchmarks."""
import os
from pathlib import Path
from textwrap import dedent

import click
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib_venn import venn2

HERE = Path(__file__).parent.resolve()
RESULTS = HERE.joinpath('results', 'bioid_performance')


@click.command()
@click.option("--reference")
@click.option("--comparison")
@click.option("--key", default="exists_correct")
def main(reference: str, comparison: str, key: str):
    df1 = pd.read_json(reference, orient="record")
    df2 = pd.read_json(comparison, orient="record")
    reference_base = os.path.splitext(os.path.basename(reference))[0]
    comparison_base = os.path.splitext(os.path.basename(comparison))[0]
    output = RESULTS.joinpath(f"{reference_base}_{comparison_base}", key)
    output.mkdir(exist_ok=True, parents=True)

    fig, ax = plt.subplots()
    venn2(
        [
            set(df1[df1[key]].index),
            set(df2[df2[key]].index),
        ],
        [
            reference_base,
            comparison_base,
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
    
    {new_idx.sum():,} success rows ({new_unique_count:,} unique) in {comparison_base} but not {reference_base}
    {regressions_idx.sum():,} success rows ({regression_unique_count:,} unique) in {reference_base} but not {comparison_base}
    {misses.sum():,} rows missed ({misses_unique_count:,} unique) in both
    {hits_idx.sum():,} successes rows ({hits_unique_count:,} unique) in both
    """))


if __name__ == '__main__':
    main()
