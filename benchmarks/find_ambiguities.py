import os
from gilda.grounder import Grounder
from gilda.resources import get_grounding_terms

grounding_terms_file = os.path.join(os.path.dirname(__file__), os.pardir,
                                    os.pardir, 'resources',
                                    'grounding_terms.tsv')

gr = Grounder(get_grounding_terms())

ambig_entries = {}
asserted_entries = []
for terms in gr.entries.values():
    for term in terms:
        if term.status == 'assertion':
            asserted_entries.append(term.text)
        if term.db != 'HGNC':
            continue
        # We consider it an ambiguity if the same text entry appears
        # multiple times in the same DB
        key = term.text
        if key in ambig_entries:
            ambig_entries[key].append(term)
        else:
            ambig_entries[key] = [term]

ambigs = []
for text, entries in ambig_entries.items():
    # We skip assertions because they are prioritized anyway
    if text in asserted_entries:
        continue
    # If there aren't multiple entries, we skip it
    if len(entries) <= 1:
        continue
    # If the entries all point to the same ID, we skip it
    ids = {e.id for e in entries}
    if len(ids) <= 1:
        continue
    # If there is a name in statuses, we skip it becasue it's prioritized
    statuses = {e.status for e in entries}
    if 'name' in statuses:
        continue
    # Everything else is an ambguity
    ambigs.append(entries)


# This function helps identify hypothetical families based on share prefixes
def lcp(strs):
    prefix = ''
    for letters in zip(*strs):
        if len(set(letters)) == 1:
            prefix += letters[0]
        else:
            break
    return prefix


# Find possible families
for ambig in ambigs:
    names = [term.entry_name for term in ambig]
    prefix = lcp(names)
    if len(prefix) < 3:
        continue
    print('%s -> %s' % (ambig[0].text, ', '.join(sorted(names))))
