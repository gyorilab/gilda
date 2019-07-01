class Term(object):
    """Represents a text entry corresponding to a grounded term.

    Attributes
    ----------
    norm_text : str
        The normalized text corresponding to the text entry, used for lookups.
    text : str
        The text entry itself.
    db : str
        The database / name space corresponding to the grounded term.
    id : str
        The identifier of the grounded term within the database / name space.
    entry_name : str
        The standardized name corresponding to the grounded term.
    status : str
        The relationship of the text entry to the grounded term, e.g., synonym.
    source : str
        The source from which the term was obtained.
    """
    def __init__(self, norm_text, text, db, id, entry_name, status, source):
        self.norm_text = norm_text
        self.text = text
        self.db = db
        self.id = str(id)
        self.entry_name = entry_name
        self.status = status
        self.source = source

    def __str__(self):
        return 'Term(%s,%s,%s,%s,%s,%s,%s)' % (
            self.norm_text, self.text, self.db, self.id, self.entry_name,
            self.status, self.source)

    def __repr__(self):
        return str(self)

    def to_json(self):
        """Return the term serialized into a JSON dict."""
        return {
            'norm_text': self.norm_text,
            'text': self.text,
            'db': self.db,
            'id': self.id,
            'entry_name': self.entry_name,
            'status': self.status,
            'source': self.source
        }

    def to_list(self):
        """Return the term serialized into a list of strings."""
        return [self.norm_text, self.text, self.db, self.id,
                self.entry_name, self.status, self.source]
