# -*- coding: utf-8 -*-

"""This script benchmarks the responsivenes (i.e., speed) of Gilda
on the BioCreative VII BioID corpus in three settings: when used as a Python
package, as a local web service, and when using the remote public web
service."""

import pathlib
import random
import time
from typing import Optional
from itertools import zip_longest

import click
import matplotlib.pyplot as plt
import pandas as pd
import requests
import seaborn as sns
from more_click import force_option, verbose_option
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

import gilda
from bioid_evaluation import BioIDBenchmarker
from gilda.api import grounder

HERE = pathlib.Path(__file__).parent.resolve()
RESULTS = HERE.joinpath("results")
RESULTS.mkdir(exist_ok=True, parents=True)

RESULTS_PATH = RESULTS.joinpath("bioid_responsiveness.tsv")
RESULTS_AGG_PATH = RESULTS.joinpath("bioid_responsiveness_aggregated.tsv")
RESULTS_AGG_TEX_PATH = RESULTS.joinpath("bioid_responsiveness_aggregated.tex")
FIG_PATH = RESULTS.joinpath("bioid_responsiveness.svg")
FIG_PDF_PATH = RESULTS.joinpath("bioid_responsiveness.pdf")


def ground_package(text, **_kwargs):
    return gilda.ground(text)


def ground_package_context(text, context):
    return gilda.ground(text, context=context)


def ground_app_local_multi(batch):
    return requests.post(
            "http://localhost:8001/ground_multi",
            json=[{'text': text} for text, _ in batch]).json()


def ground_app_local_multi_context(batch):
    return requests.post(
            "http://localhost:8001/ground_multi",
            json=[{'text': text, 'context': context}
                   for text, context in batch]).json()


def ground_app_local(text, **_kwargs):
    return requests.post("http://localhost:8001/ground", json={"text": text}).json()


def ground_app_local_context(text, context):
    return requests.post(
        "http://localhost:8001/ground", json={"text": text, "context": context}
    ).json()

def ground_app_remote(text, **_kwargs):
    return requests.post(
        "http://grounding.indra.bio/ground", json={"text": text}
    ).json()


def ground_app_remote_context(text, context):
    return requests.post(
        "http://grounding.indra.bio/ground", json={"text": text, "context": context}
    ).json()


#: A list of benchmarks to run with three columns:
#:  type, uses context, function
FUNCTIONS = [
    ("Python package", False, ground_package, None),
    ("Python package", True, ground_package_context, None),
    ("Local web app", False, ground_app_local, None),
    ("Local web app", True, ground_app_local_context, None),
    ("Local web app multi", False, ground_app_local_multi, 100),
    ("Local web app multi", True, ground_app_local_multi_context, 100),
    ("Public web app", False, ground_app_remote, None),
    ("Public web app", True, ground_app_remote_context, None),
]


def run_trial(
    *, trials, corpus, func, desc: Optional[str] = None, chunk: Optional[int] = None,
    batching: Optional[int] = None
):
    rv = []
    outer_it = trange(trials, desc=desc)
    for trial in outer_it:
        random.shuffle(corpus)
        test_corpus = corpus[:chunk] if chunk else corpus
        if batching:
            it = zip_longest(*[iter(test_corpus)]*batching, fillvalue=None)
            for batch in tqdm(it, desc="Examples", unit_scale=True, leave=False):
                items = [x for x in batch if x is not None]
                with logging_redirect_tqdm():
                    start = time.time()
                    all_matches = func(items)
                    for matches in all_matches:
                        rv.append((trial, len(matches),
                                   (time.time() - start)/len(items)))
        else:
            for text, context in tqdm(test_corpus, desc="Examples", unit_scale=True, leave=False):
                with logging_redirect_tqdm():
                    start = time.time()
                    matches = func(text, context=context)
                    rv.append((trial, len(matches), time.time() - start))
    return rv


def iter_corpus():
    benchmarker = BioIDBenchmarker()
    for text, article in tqdm(benchmarker.processed_data[['text', 'don_article']].values):
        yield text, benchmarker._get_plaintext(article)


def build(trials: int, chunk: Optional[int] = None) -> pd.DataFrame:
    click.secho("Warming up python grounder")
    start = time.time()
    grounder.get_grounder()
    end = time.time() - start
    click.secho(f"Warmed up in {end:.2f} seconds")

    click.secho("Warming up local api grounder")
    ground_app_local_context("ER", context="Calcium is released from the ER.")

    click.secho("Warming up remote api grounder")
    ground_app_remote_context("ER", context="Calcium is released from the ER.")

    click.secho("Preparing BioID corpus")
    corpus = list(iter_corpus())

    rows = []
    for tag, uses_context, func, batching in FUNCTIONS:
        rv = run_trial(
            trials=trials,
            chunk=chunk,
            corpus=corpus,
            func=func,
            batching=batching,
            desc=f"{tag}{' with context' if uses_context else ''} trial",
        )
        rows.extend((tag, uses_context, *row) for row in rv)
    df = pd.DataFrame(rows, columns=["type", "context", "trial", "matches", "duration"])
    return df


@click.command()
@click.option("--trials", type=int, default=2, show_default=True)
@click.option(
    "--chunk",
    type=int,
    help="Subsample size from full corpus. Defaults to full corpus if not given.",
)
@verbose_option
@force_option
def main(trials: int, chunk: Optional[int], force: bool):
    if RESULTS_PATH.is_file() and not force:
        df = pd.read_csv(RESULTS_PATH, sep="\t")
    else:
        df = build(trials=trials, chunk=chunk)
        df.to_csv(RESULTS_PATH, sep="\t", index=False)

    # convert from seconds/response to responses/second
    df["duration"] = df["duration"].map(lambda x: 1 / x)

    _grouped = df[["type", "context", "duration"]].groupby(["type", "context"])
    agg_mean_df = _grouped.mean()
    agg_mean_df.rename(columns={"duration": "duration_mean"}, inplace=True)
    agg_std_df = _grouped.std()
    agg_std_df.rename(columns={"duration": "duration_std"}, inplace=True)
    agg_df = pd.merge(agg_mean_df, agg_std_df, left_index=True, right_index=True)
    agg_df = agg_df.round(1)
    agg_df.to_csv(RESULTS_AGG_PATH, sep="\t")
    agg_df.to_latex(
        RESULTS_AGG_TEX_PATH,
        label="tab:bioid-responsiveness-benchmark",
    )

    fig, ax = plt.subplots(figsize=(6, 3))
    sns.boxplot(data=df, y="duration", x="type", hue="context", ax=ax)
    ax.set_title("Gilda Responsiveness Benchmark on BioID")
    ax.set_yscale("log")
    ax.set_ylabel("Responses per Second")
    ax.set_xlabel("")
    fig.savefig(FIG_PATH)
    fig.savefig(FIG_PDF_PATH)


if __name__ == "__main__":
    main()
