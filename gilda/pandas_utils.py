"""Utilities for Pandas."""

from functools import partial
from typing import Optional, Union, TYPE_CHECKING

from .grounder import Grounder
from . import api

if TYPE_CHECKING:
    import pandas

__all__ = [
    "ground_df",
    "ground_df_map",
]


def ground_df(
    df: "pandas.DataFrame",
    source_column: Union[str, int],
    *,
    target_column: Union[None, str, int] = None,
    grounder: Optional[Grounder] = None,
    **kwargs,
) -> None:
    """
    Ground the elements of a column in a Pandas dataframe as CURIEs, in-place.

    Parameters
    ----------
    df :
        A pandas dataframe
    source_column :
        The column to ground. This column contains text corresponding
        to named entities' labels or synonyms
    target_column :
        The column where to put the groundings (either a CURIE string,
        or None). It's possible to create a new column when passing
        a string for this argument. If not given, will create a new
        column name like ``<source column>_grounded``.
    grounder :
        A custom grounder. If none given, uses the built-in grounder.
    kwargs :
        Keyword arguments passed to :meth:`Grounder.ground`, could
        include context, organisms, or namespaces.

    Examples
    --------
    The following example shows how to use this function.

    .. code-block:: python

        import pandas as pd
        import gilda

        url = "https://raw.githubusercontent.com/OBOAcademy/obook/master/docs/tutorial/linking_data/data.csv"
        df = pd.read_csv(url)
        gilda.ground_df(df, source_column="disease", target_column="disease_curie")
    """
    if target_column is None:
        target_column = f"{source_column}_grounded"
    df[target_column] = ground_df_map(
        df=df, source_column=source_column, grounder=grounder, **kwargs,
    )


def ground_df_map(
    df: "pandas.DataFrame",
    source_column: Union[str, int],
    *,
    grounder: Optional[Grounder] = None,
    **kwargs,
) -> "pandas.Series":
    """
    Ground the elements of a column in a Pandas dataframe as CURIEs.

    Parameters
    ----------
    df :
        A pandas dataframe
    source_column :
        The column to ground. This column contains text corresponding
        to named entities' labels or synonyms
    grounder :
        A custom grounder. If none given, uses the built-in ground.
    kwargs :
        Keyword arguments passed to :meth:`Grounder.ground`, could
        include context, organisms, or namespaces.

    Returns
    -------
    series :
        A pandas series representing the grounded CURIE strings.
        Contains NaNs if grounding was not successful or if
        there was an NaN in the cell before.
    """
    if grounder is None:
        grounder = api.grounder
    func = partial(_ground_helper, grounder=grounder, **kwargs)
    series = df[source_column].map(func)
    return series


def _ground_helper(text, grounder: Grounder, **kwargs) -> Optional[str]:
    if not isinstance(text, str):
        return None
    scored_matches = grounder.ground(text, **kwargs)
    if not scored_matches:
        return None
    return scored_matches[0].term.get_curie()
