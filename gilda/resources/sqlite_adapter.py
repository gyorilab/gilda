"""This module implements an optional SQLite database back-end for Gilda
as an alternative to loading Terms directly into memory."""

import os
import json
import sys
import logging
import sqlite3
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
        self.conn = None

    def get_connection(self):
        if self.conn:
            return self.conn
        self.conn = sqlite3.connect(self.db)
        return self.conn

    def get(self, key, default=None):
        res = self.get_connection().execute(
            "SELECT terms FROM terms WHERE norm_text=?", (key,))
        result = res.fetchone()
        if not result:
            return default
        return [Term(**j) for j in json.loads(result[0])]

    def values(self):
        res = self.get_connection().execute("SELECT terms FROM terms")
        for result in res.fetchall():
            yield [Term(**j) for j in json.loads(result[0])]

    def __getitem__(self, item):
        res = self.get(item)
        if res is None:
            raise KeyError(item)
        return res

    def __len__(self):
        res = self.get_connection().execute("SELECT COUNT(norm_text) FROM terms")
        return res.fetchone()[0]

    def __iter__(self):
        res = self.get_connection().execute("SELECT norm_text FROM terms")
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

    # Create the table
    logger.info('Creating the table')
    q = "CREATE TABLE terms (norm_text text not null primary key, terms text)"
    cur.execute(q)

    # Insert terms
    logger.info('Inserting terms')
    q = "INSERT INTO terms (norm_text, terms) VALUES (?, ?)"
    for norm_text, terms in grounding_entries.items():
        cur.execute(q, (norm_text, json.dumps([t.to_json() for t in terms])))

    # Build index
    logger.info('Making index')
    q = "CREATE INDEX norm_index ON terms (norm_text);"
    cur.execute(q)
    conn.commit()
    conn.close()


if __name__ == '__main__':
    from gilda.grounder import Grounder

    path = sys.argv[1] if len(sys.argv) > 1 else None
    logger.info('Loading default grounder')
    gr = Grounder()
    build(gr.entries, path=path)
