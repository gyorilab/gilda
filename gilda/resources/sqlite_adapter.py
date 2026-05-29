"""This module implements an optional SQLite database back-end for Gilda
as an alternative to loading Terms directly into memory.

It uses a flat relational table with one row per Term and one column per
Term attribute. The set of columns is derived from
the ``Term``constructor signature so that the schema follows the Term
model.
"""

import os
import sys
import inspect
import logging
import sqlite3
import itertools
import threading
from gilda.term import Term
from . import resource_dir

logger = logging.getLogger('gilda.resources.sqlite_adapter')


class SqliteEntries:
    """A class exposing lists of Terms similar to a string-keyed dict.

    From the perspective of a Grounder instance, instances of this class
    have an interface similar to a dict ot lists of Terms and can therefore
    be used seamlessly as a Grounder instance's entries attribute.

    Parameters
    ----------
    db : str
        A path to a SQLite database file.
    """
    def __init__(self, db):
        self.db = db
        self._local = threading.local()
        self._columns = None
        self._cols_clause = None

    @property
    def conn(self):
        return self.get_connection()

    def get_connection(self):
        # Use hasattr rather than checking for None because each thread has its
        # own local instance of threading.local, which may or may not have the
        # conn attribute set
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(self.db)
        return self._local.conn

    @property
    def columns(self):
        """The Term columns present in the database, in definition order."""
        if self._columns is None:
            res = self.get_connection().execute("PRAGMA table_info(terms)")
            self._columns = [row[1] for row in res.fetchall()]
        return self._columns

    @property
    def _column_clause(self):
        """The comma-separated column list for SELECTs, computed once."""
        if self._cols_clause is None:
            self._cols_clause = ', '.join(self.columns)
        return self._cols_clause

    def get(self, key, default=None):
        q = "SELECT %s FROM terms WHERE norm_text=?" % self._column_clause
        rows = self.get_connection().execute(q, (key,)).fetchall()
        if not rows:
            return default
        return [Term(*row) for row in rows]

    def values(self):
        nt_idx = self.columns.index('norm_text')
        q = "SELECT %s FROM terms ORDER BY norm_text" % self._column_clause
        res = self.get_connection().execute(q)
        for _, group in itertools.groupby(res, key=lambda row: row[nt_idx]):
            yield [Term(*row) for row in group]

    def __getitem__(self, item):
        res = self.get(item)
        if res is None:
            raise KeyError(item)
        return res

    def __len__(self):
        res = self.get_connection().execute(
            "SELECT COUNT(DISTINCT norm_text) FROM terms")
        return res.fetchone()[0]

    def __iter__(self):
        res = self.get_connection().execute(
            "SELECT DISTINCT norm_text FROM terms")
        for norm_text, in res.fetchall():
            yield norm_text


def build(grounding_entries, path=None):
    """Build a SQLite database file from a set of grounding entries.

    Parameters
    ----------
    grounding_entries : dict[str, list[Term]]
        A grounding entries data structure from which the DB is generated.
    path : Optional[str, Path]
        Optional path to the output file which should use the .db extension.
        If not given, the .db file is generated in Gilda's default resources
        folder.
    """
    path = path if path else os.path.join(resource_dir, 'grounding_terms.db')
    logger.info('Starting SQLite database at %s' % path)
    conn = sqlite3.connect(path)
    cur = conn.cursor()

    # Derive the schema from the Term model rather than hard-coding it
    cols = _term_columns()

    # Create the table with one column per Term attribute
    logger.info('Creating the table with columns: %s' % ', '.join(cols))
    col_defs = ', '.join('%s text' % col for col in cols)
    cur.execute("CREATE TABLE terms (%s)" % col_defs)

    # Insert one row per Term
    logger.info('Inserting terms')
    placeholders = ', '.join(['?'] * len(cols))
    q = "INSERT INTO terms (%s) VALUES (%s)" % (', '.join(cols), placeholders)

    def row_generator():
        for norm_text, terms in grounding_entries.items():
            for term in terms:
                yield tuple(getattr(term, col) for col in cols)

    cur.executemany(q, row_generator())

    # Build index
    logger.info('Making index')
    cur.execute("CREATE INDEX norm_index ON terms (norm_text);")
    conn.commit()
    conn.close()


def _term_columns():
    """Return the Term attribute names derived from its constructor."""
    params = inspect.signature(Term.__init__).parameters
    return [name for name in params if name != 'self']


if __name__ == '__main__':
    from gilda.grounder import Grounder

    path = sys.argv[1] if len(sys.argv) > 1 else None
    logger.info('Loading default grounder')
    gr = Grounder()
    build(gr.entries, path=path)
