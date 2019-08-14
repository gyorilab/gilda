from gilda.grounder import Grounder
from gilda.resources import get_grounding_terms
from indra.databases import mesh_client

gr = Grounder(get_grounding_terms())

mesh_protein = 'D000602'
mesh_enzyme = 'D045762'


def dump_mappings(mappings, fname):
    with open(fname, 'w') as fh:
        for me, he in mappings:
            fh.write('\t'.join(['MESH', me.id, 'HGNC', he.id]) + '\n')


def get_mesh_hgnc_mappings(ambigs):
    predicted_mappings = []
    for _, ambig in ambigs.items():
        hgnc_entries = [a for a in ambig if a.db == 'HGNC']
        mesh_entries = [a for a in ambig if a.db == 'MESH']
        if len(hgnc_entries) != 1 or len(mesh_entries) != 1:
            continue
        he = hgnc_entries[0]
        me = mesh_entries[0]

        if mesh_client.mesh_isa(me.id, mesh_protein) or \
                mesh_client.mesh_isa(me.id, mesh_enzyme):
            predicted_mappings.append((me, he))
    return predicted_mappings


def find_ambiguities(gr):
    ambig_entries = {}
    for terms in gr.entries.values():
        for term in terms:
            # We consider it an ambiguity if the same text entry appears
            # multiple times
            key = term.text
            if key in ambig_entries:
                ambig_entries[key].append(term)
            else:
                ambig_entries[key] = [term]
    # It's only an ambiguity if there are two entries at least
    ambig_entries = {k: v for k, v in ambig_entries.items() if len(v) >= 2}
    return ambig_entries


if __name__ == '__main__':
    ambigs = find_ambiguities(gr)
    mappings = get_mesh_hgnc_mappings(ambigs)
    dump_mappings(mappings, 'mesh_mappings.tsv')