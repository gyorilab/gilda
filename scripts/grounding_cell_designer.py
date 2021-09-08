"""This script can process a set of Cell Designer XML files that are
part of the COVID-19 Disease Map project, detect species with missing
grounding, ground these based on their string name with Gilda, and
serialize the changes back into XML files."""

import re
import itertools
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

import click
from indra.databases import identifiers
from tqdm import tqdm

import gilda

rdf_str = (
    b'<rdf:RDF '
    b'xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" '
    b'xmlns:bqmodel="http://biomodels.net/model-qualifiers/" '
    b'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    b'xmlns:dcterms="http://purl.org/dc/terms/" '
    b'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    b'xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#">'
)

namespaces = {'sbml': 'http://www.sbml.org/sbml/level2/version4',
              'celldesigner': 'http://www.sbml.org/2001/ns/celldesigner',
              'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
              'bqbiol': 'http://biomodels.net/biology-qualifiers/',
              'bqmodel': 'http://biomodels.net/model-qualifiers/',
              'vCard': 'http://www.w3.org/2001/vcard-rdf/3.0#',
              'dc': 'http://purl.org/dc/elements/1.1/',
              'xhtml': 'http://www.w3.org/1999/xhtml'}


irrelevant_grounding_ns = {'pubmed', 'taxonomy', 'doi', 'wikipathways', 'pdb',
                           'intact', 'biogrid', 'pmc'}
relevant_grounding_ns = {'ncbiprotein', 'ncbigene', 'uniprot', 'obo.go',
                         'obo.chebi', 'pubchem.compound', 'pubchem.substance',
                         'hgnc', 'hgnc.symbol', 'mesh', 'interpro',
                         'refseq', 'ensembl', 'ec-code', 'brenda',
                         'kegg.compound', 'drugbank'}
irrelevant_classes = {'DEGRADED'}
grounding_stats = {'ngrounding': 0}


def register_all_namespaces(fname):
    namespaces = dict([node for _, node in
                       ET.iterparse(fname, events=['start-ns'])])
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])


def get_protein_reference(species):
    protein_ref = species.find('sbml:annotation/'
                               'celldesigner:extension/'
                               'celldesigner:speciesIdentity/'
                               'celldesigner:proteinReference',
                               namespaces=namespaces)
    if protein_ref is None:
        protein_ref = species.find('celldesigner:annotation/'
                                   'celldesigner:speciesIdentity/'
                                   'celldesigner:proteinReference',
                                   namespaces=namespaces)
        if protein_ref is None:
            protein_ref = species.find('celldesigner:extension/'
                                       'celldesigner:speciesIdentity/'
                                       'celldesigner:proteinReference',
                                       namespaces=namespaces)

    if protein_ref is not None:
        protein_ref_id = protein_ref.text
        return protein_ref_id


def add_groundings(et):
    all_species = et.findall('sbml:model/sbml:listOfSpecies/sbml:species',
                             namespaces=namespaces)
    print('%d species found with first method' % len(all_species))
    all_species += et.findall('sbml:model/sbml:annotation/'
                              'celldesigner:extension/'
                              'celldesigner:listOfIncludedSpecies/'
                              'celldesigner:species',
                              namespaces=namespaces)
    print('%d species found after second method' % len(all_species))
    # This is to collect protein reference-based groundings first
    groundings_for_prot_ref = defaultdict(set)
    species_for_prot_ref = defaultdict(set)
    for species in all_species:
        protein_ref_id = get_protein_reference(species)
        if protein_ref_id:
            existing_grounding = get_existing_grounding(species)
            groundings_for_prot_ref[protein_ref_id] |= \
                set(existing_grounding) & relevant_grounding_ns
            species_for_prot_ref[protein_ref_id].add(species.attrib['id'])
    for species in tqdm(all_species, desc='Species', leave=False):
        # Skip if we have a grounding via the protein reference, indirectly
        protein_ref_id = get_protein_reference(species)
        if protein_ref_id and groundings_for_prot_ref.get(protein_ref_id):
            continue
        existing_grounding = get_existing_grounding(species)
        # Important: this is where we decide if we will add any grounding.
        # Here we skip this species if it has any relevant grounding
        if set(existing_grounding) & relevant_grounding_ns:
            continue
        class_tag = species.find(
            'sbml:annotation/celldesigner:extension/'
            'celldesigner:speciesIdentity/celldesigner:class',
            namespaces=namespaces)
        if class_tag is not None:
            entity_class = class_tag.text
            if entity_class in irrelevant_classes:
                continue
        name = species.attrib.get('name')
        tqdm.write(name)
        tqdm.write(entity_class)
        matches = gilda.ground(name)
        if matches:
            species = add_grounding_element(species, entity_class,
                                            matches[0].term.db,
                                            matches[0].term.id)
            if species:
                tqdm.write(' '.join((name, matches[0].term.db, matches[0].term.id,
                      matches[0].term.entry_name)))
                grounding_stats['ngrounding'] += 1
        tqdm.write('---')
    return et


def get_existing_grounding(species):
    # Others: isHomologTo
    bqbio_tags = ['isDescribedBy', 'isEncodedBy', 'is', 'encodes',
                  'occursIn']
    prefixes = ['sbml:annotation', 'celldesigner:notes/xhtml:html/xhtml:body']

    groundings = defaultdict(list)
    for prefix, tag in itertools.product(prefixes, bqbio_tags):
        grounding_elements = \
            species.findall('%s/rdf:RDF/rdf:Description/'
                            'bqbiol:%s/rdf:Bag/rdf:li' % (prefix, tag),
                            namespaces=namespaces)
        for element in grounding_elements:
            urn = element.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
                                 'resource']
            match = re.match(r'urn:miriam:([^:]+):(.+)', urn)
            if not match:
                tqdm.write(f'Unmatched urn: {urn}')
                continue
            else:
                db_ns, db_id = match.groups()
                groundings[db_ns].append(db_id)
    return groundings


def add_grounding_element(species, entity_class, db_ns, db_id):
    # For genes, if we're grounding a protein, we make the encoding aspect
    # explicit
    if entity_class == 'PROTEIN' and db_ns == 'HGNC':
        bqbiol_tag = 'bqbiol:isEncodedBy'
    # In case a protein is grounded to CHEBI, it's typically a problem, we
    # skip these
    elif entity_class == 'PROTEIN' and db_ns == 'CHEBI':
        return
    else:
        bqbiol_tag = 'bqbiol:is'

    tag_sequence = [
        'sbml:annotation',
        'rdf:RDF',
        'rdf:Description',
        bqbiol_tag,
        'rdf:Bag',
    ]

    identifiers_ns = identifiers.get_identifiers_ns(db_ns)
    grounding_str = 'urn:miriam:%s:%s' % (identifiers_ns, db_id)

    root = species
    for tag in tag_sequence:
        element = root.find(tag, namespaces=namespaces)
        if element is not None:
            root = element
        else:
            new_element = ET.Element(tag)
            new_element.text = '\n'
            new_element.tail = '\n'
            root.append(new_element)
            root = new_element

    li = ET.Element('rdf:li', attrib={'rdf:resource': grounding_str})
    root.append(li)
    return species


def dump(et, fname):
    xml_str = ET.tostring(et.getroot(), xml_declaration=False,
                          encoding='UTF-8')
    xml_str = b'<?xml version="1.0" encoding="UTF-8"?>\n' + xml_str
    xml_str = xml_str.replace(b'ns0:', b'')
    xml_str = xml_str.replace(b'ns1:', b'')
    xml_str = xml_str.replace(b' />', b'/>')
    xml_str = xml_str.replace(b'<html>',
                              b'<html xmlns="http://www.w3.org/1999/xhtml">')
    xml_str = xml_str.replace(b'<rdf:RDF>', rdf_str)
    xml_str = xml_str.replace(b'xmlns:ns0="http://www.sbml.org/sbml/level2/version4"',
                              b'')
    xml_str = xml_str.replace(b'xmlns:ns1="http://www.w3.org/1999/xhtml"', b'')
    with open(fname, 'wb') as fh:
        fh.write(xml_str)


@click.command()
@click.argument('directory', type=Path)
def main(directory: Path):
    """Run grounding on the directory for the COVID 19 Disease Maps repository."""
    stable_xmls = list(directory.resolve().rglob('*_stable.xml'))
    it = tqdm(stable_xmls, desc='Grounding')
    for stable_xml in it:
        it.set_postfix(file=stable_xml.name)
        register_all_namespaces(stable_xml)
        et = ET.parse(stable_xml)
        et = add_groundings(et)
        dump(et, stable_xml)


if __name__ == '__main__':
    gilda.ground('x')
    main()
