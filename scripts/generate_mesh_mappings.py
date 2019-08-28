from gilda.grounder import Grounder
from gilda.resources import get_grounding_terms
from indra.databases import mesh_client, hgnc_client

gr = Grounder(get_grounding_terms())

mesh_protein = 'D000602'
mesh_enzyme = 'D045762'


def dump_mappings(mappings, fname):
    mappings = sorted(mappings.values(), key=lambda x: x[0].id)
    with open(fname, 'w') as fh:
        for me, te in mappings:
            mesh_name = mesh_client.get_mesh_name(me.id)
            if te.db == 'HGNC':
                tname = hgnc_client.get_hgnc_name(te.id)
            elif te.db == 'FPLX':
                tname = te.id
            fh.write('\t'.join([me.db, me.id, mesh_name,
                                te.db, te.id, tname]) + '\n')


def get_mesh_mappings(ambigs):
    predicted_mappings = {}
    for _, ambig in ambigs.items():
        hgnc_entries = [a for a in ambig if a.db == 'HGNC']
        mesh_entries = [a for a in ambig if a.db == 'MESH']
        fplx_entries = [a for a in ambig if a.db == 'FPLX']
        if len(mesh_entries) != 1:
            continue
        me = mesh_entries[0]
        if (mesh_client.mesh_isa(me.id, mesh_protein) or
                mesh_client.mesh_isa(me.id, mesh_enzyme)):
            if len(fplx_entries) == 1:
                key = (me.id, 'FPLX', fplx_entries[0].id)
                predicted_mappings[key] = (me, fplx_entries[0])
            elif len(hgnc_entries) == 1:
                key = (me.id, 'HGNC', hgnc_entries[0].id)
                predicted_mappings[key] = (me, hgnc_entries[0])
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
    mappings = get_mesh_mappings(ambigs)
    dump_mappings(mappings, 'mesh_mappings.tsv')