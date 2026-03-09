"""Annotate a CSV file with Gilda groundings – built-in namespaces or custom ontologies.

Usage
-----
Built-in namespace (default: CHEBI):
    python scripts/annotate_csv_chebi.py input.csv --column chemical_name
    python scripts/annotate_csv_chebi.py input.csv --column chemical_name --namespace MESH

Custom OBO file (URL or local path):
    python scripts/annotate_csv_chebi.py input.csv --column tissue \\
        --obo-url http://purl.obolibrary.org/obo/bto.obo --obo-prefix BTO

Custom OBO Graph JSON (e.g. MONDO):
    python scripts/annotate_csv_chebi.py input.csv --column disease \\
        --obograph-url https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.json \\
        --obo-prefix MONDO --uri-prefix http://purl.obolibrary.org/obo/MONDO_

With verbose output (matched name + score):
    python scripts/annotate_csv_chebi.py input.csv --column chemical_name --verbose

With a context column for disambiguation:
    python scripts/annotate_csv_chebi.py input.csv --column chemical_name --context-column sentence
"""

import argparse

import pandas as pd

import gilda
import gilda.term
from gilda import make_grounder
from gilda.process import normalize


# ---------------------------------------------------------------------------
# Grounder builders for custom ontologies
# ---------------------------------------------------------------------------

def _grounder_from_obo(url_or_path: str, prefix: str):
    """Build a Gilda grounder from a local or remote OBO file using obonet."""
    try:
        import obonet
    except ImportError:
        raise ImportError("Install obonet to use --obo-url: pip install obonet")

    print(f"Loading OBO from: {url_or_path}")
    g = obonet.read_obo(url_or_path)
    terms = []
    for node, data in g.nodes(data=True):
        if not node.startswith(f"{prefix}:"):
            continue
        identifier = node.removeprefix(f"{prefix}:")
        name = data.get("name")
        if not name:
            continue

        terms.append(gilda.term.Term(
            norm_text=normalize(name),
            text=name,
            db=prefix,
            id=identifier,
            entry_name=name,
            status="name",
            source=prefix,
        ))
        for synonym_raw in data.get("synonym", []):
            synonym = synonym_raw.split('"')[1].strip()
            terms.append(gilda.term.Term(
                norm_text=normalize(synonym),
                text=synonym,
                db=prefix,
                id=identifier,
                entry_name=name,
                status="synonym",
                source=prefix,
            ))

    print(f"Loaded {len(terms)} terms from {prefix}")
    return make_grounder(terms)


def _grounder_from_obograph(url: str, prefix: str, uri_prefix: str):
    """Build a Gilda grounder from an OBO Graph JSON file (e.g. MONDO)."""
    try:
        import requests
    except ImportError:
        raise ImportError("Install requests to use --obograph-url: pip install requests")

    print(f"Loading OBO Graph JSON from: {url}")
    res = requests.get(url)
    res.raise_for_status()
    data = res.json()
    terms = []
    for node in data["graphs"][0]["nodes"]:
        uri = node["id"]
        if not uri.startswith(uri_prefix):
            continue
        identifier = uri[len(uri_prefix):]
        name = node.get("lbl")
        if not name:
            continue

        terms.append(gilda.term.Term(
            norm_text=normalize(name),
            text=name,
            db=prefix,
            id=identifier,
            entry_name=name,
            status="name",
            source=prefix,
        ))
        for syn in node.get("meta", {}).get("synonyms", []):
            synonym = syn["val"]
            terms.append(gilda.term.Term(
                norm_text=normalize(synonym),
                text=synonym,
                db=prefix,
                id=identifier,
                entry_name=name,
                status="synonym",
                source=prefix,
            ))

    print(f"Loaded {len(terms)} terms from {prefix}")
    return make_grounder(terms)


def _build_grounder(args):
    """Return (grounder_or_None, namespace_or_None, prefix) based on CLI args."""
    if args.obo_url:
        if not args.obo_prefix:
            raise ValueError("--obo-prefix is required when using --obo-url")
        return _grounder_from_obo(args.obo_url, args.obo_prefix), None, args.obo_prefix

    if args.obograph_url:
        if not args.obo_prefix:
            raise ValueError("--obo-prefix is required when using --obograph-url")
        if not args.uri_prefix:
            raise ValueError("--uri-prefix is required when using --obograph-url")
        return (
            _grounder_from_obograph(args.obograph_url, args.obo_prefix, args.uri_prefix),
            None,
            args.obo_prefix,
        )

    # Built-in namespace via the default Gilda grounder
    ns = args.namespace or "CHEBI"
    return None, ns, ns


# ---------------------------------------------------------------------------
# Per-row grounding helper
# ---------------------------------------------------------------------------

def _ground_row(text, context=None, grounder=None, namespace=None):
    """Ground one text value; return (curie, entry_name, score)."""
    if not isinstance(text, str):
        return None, None, None

    if grounder is not None:
        matches = grounder.ground(text, context=context)
    else:
        matches = gilda.ground(text, context=context, namespaces=[namespace])

    if not matches:
        return None, None, None
    top = matches[0]
    return top.term.get_curie(), top.term.entry_name, round(top.score, 4)


# ---------------------------------------------------------------------------
# Main annotation function
# ---------------------------------------------------------------------------

def annotate_csv(
    input_path: str,
    column: str,
    output_path: str,
    context_column: str = None,
    verbose: bool = False,
    grounder=None,
    namespace: str = None,
    prefix: str = "result",
) -> pd.DataFrame:
    """Load a CSV, annotate one column with Gilda groundings, and write the result.

    Parameters
    ----------
    input_path:
        Path to the input CSV file.
    column:
        Name of the column containing entity text to ground.
    output_path:
        Path where the annotated CSV will be saved.
    context_column:
        Optional column with sentence context for disambiguation.
    verbose:
        If True, also adds ``<prefix>_name`` and ``score`` columns.
    grounder:
        A custom Gilda Grounder (from make_grounder). Mutually exclusive with
        ``namespace``.
    namespace:
        A built-in Gilda namespace string (e.g. ``"CHEBI"``, ``"MESH"``).
    prefix:
        Short label used to name the output columns (e.g. ``"chebi"`` →
        ``chebi_curie``, ``chebi_name``).
    """
    df = pd.read_csv(input_path)

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' not found. Available: {list(df.columns)}"
        )
    if context_column and context_column not in df.columns:
        raise ValueError(
            f"Context column '{context_column}' not found. Available: {list(df.columns)}"
        )

    ns_label = namespace or prefix
    print(f"Grounding {len(df)} rows from '{column}' against {ns_label}...")

    curie_col = f"{prefix.lower()}_curie"

    def _apply(row):
        ctx = row[context_column] if context_column else None
        return pd.Series(_ground_row(row[column], context=ctx, grounder=grounder, namespace=namespace))

    df[[curie_col, f"{prefix.lower()}_name", "score"]] = df.apply(_apply, axis=1)

    if not verbose:
        df.drop(columns=[f"{prefix.lower()}_name", "score"], inplace=True)

    df.to_csv(output_path, index=False)

    matched = df[curie_col].notna().sum()
    total = len(df)
    pct = 100 * matched // total if total else 0
    print(f"Matched {matched}/{total} rows ({pct}%)")
    print(f"Saved annotated CSV to: {output_path}")
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Annotate a CSV column with Gilda groundings.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the input CSV file.")
    parser.add_argument("--column", required=True, help="Column with entity text to ground.")
    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path. Defaults to <input>_annotated.csv.",
    )
    parser.add_argument(
        "--context-column",
        default=None,
        help="Column with sentence context for disambiguation.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also output matched name and score columns.",
    )

    # --- Built-in namespace mode ---
    parser.add_argument(
        "--namespace",
        default=None,
        metavar="NS",
        help="Built-in Gilda namespace to ground against (e.g. CHEBI, MESH, GO). "
             "Default: CHEBI.",
    )

    # --- Custom OBO mode ---
    parser.add_argument(
        "--obo-url",
        default=None,
        metavar="URL",
        help="URL or local path to an OBO file (requires obonet). "
             "Use with --obo-prefix.",
    )

    # --- Custom OBO Graph JSON mode ---
    parser.add_argument(
        "--obograph-url",
        default=None,
        metavar="URL",
        help="URL to an OBO Graph JSON file (e.g. MONDO). "
             "Use with --obo-prefix and --uri-prefix.",
    )
    parser.add_argument(
        "--uri-prefix",
        default=None,
        metavar="URI",
        help="URI prefix for filtering nodes in OBO Graph JSON "
             "(e.g. http://purl.obolibrary.org/obo/MONDO_).",
    )

    # --- Shared for custom modes ---
    parser.add_argument(
        "--obo-prefix",
        default=None,
        metavar="PREFIX",
        help="Ontology prefix used as the namespace/CURIE prefix (e.g. BTO, MONDO).",
    )

    args = parser.parse_args()

    if args.obo_url and args.obograph_url:
        parser.error("Use either --obo-url or --obograph-url, not both.")

    grounder, namespace, prefix = _build_grounder(args)

    output_path = args.output
    if output_path is None:
        stem = args.input.removesuffix(".csv")
        output_path = f"{stem}_annotated.csv"

    annotate_csv(
        input_path=args.input,
        column=args.column,
        output_path=output_path,
        context_column=args.context_column,
        verbose=args.verbose,
        grounder=grounder,
        namespace=namespace,
        prefix=prefix.lower() if prefix else "result",
    )


if __name__ == "__main__":
    main()
