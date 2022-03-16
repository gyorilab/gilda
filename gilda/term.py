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
    organism : Optional[str]
        When the term represents a protein, this attribute provides the
        taxonomy code of the species for the protein.
        For non-proteins, not provided. Default: None
    source_db : Optional[str]
        If the term's db/id was mapped from a different, original db/id
        from a given source, this attribute provides the original db value
        before mapping.
    source_id : Optional[str]
        If the term's db/id was mapped from a different, original db/id
        from a given source, this attribute provides the original ID value
        before mapping.
    """
    def __init__(self, norm_text, text, db, id, entry_name, status, source,
                 organism=None, source_db=None, source_id=None):
        if not text:
            raise ValueError('Text for Term cannot be empty')
        self.norm_text = norm_text
        self.text = text
        self.db = db
        self.id = str(id)
        self.entry_name = entry_name
        self.status = status
        self.source = source
        self.organism = organism
        self.source_db = source_db
        self.source_id = source_id

    def __str__(self):
        return 'Term(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)' % (
            self.norm_text, self.text, self.db, self.id, self.entry_name,
            self.status, self.source, self.organism, self.source_db,
            self.source_id)

    def __repr__(self):
        return str(self)

    def to_json(self):
        """Return the term serialized into a JSON dict."""
        js = {
            'norm_text': self.norm_text,
            'text': self.text,
            'db': self.db,
            'id': self.id,
            'entry_name': self.entry_name,
            'status': self.status,
            'source': self.source,
        }
        if self.organism:
            js['organism'] = self.organism
        if self.source_db:
            js['source_db'] = self.source_db
        if self.source_id:
            js['source_id'] = self.source_id
        return js

    def to_list(self):
        """Return the term serialized into a list of strings."""
        return [self.norm_text, self.text, self.db, self.id,
                self.entry_name, self.status, self.source,
                self.organism, self.source_db, self.source_id]

    def get_idenfiers_url(self):
        return get_identifiers_url(self.db, self.id)


def get_identifiers_url(db, id):
    url_pattern = 'https://identifiers.org/{db}:{id}'
    if db == 'UP':
        db = 'uniprot'
    id_parts = id.split(':')
    if len(id_parts) == 1:
        return url_pattern.format(db=db.lower(), id=id)
    elif len(id_parts) == 2:
        return url_pattern.format(db=id_parts[0].upper(), id=id_parts[-1])
