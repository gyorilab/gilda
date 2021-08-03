import re
import sys
from collections import defaultdict
import gilda
from pathlib import Path
from xml.etree import ElementTree as ET
from indra.databases import identifiers

rdf_str = (b'<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
           b'xmlns:dc="http://purl.org/dc/elements/1.1/" '
           b'xmlns:dcterms="http://purl.org/dc/terms/" '
           b'xmlns:vCard="http://www.w3.org/2001/vcard-rdf/3.0#" '
           b'xmlns:bqbiol="http://biomodels.net/biology-qualifiers/" '
           b'xmlns:bqmodel="http://biomodels.net/model-qualifiers/">')

namespaces = {'sbml': 'http://www.sbml.org/sbml/level2/version4',
              'celldesigner': 'http://www.sbml.org/2001/ns/celldesigner',
              'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
              'bqbiol': 'http://biomodels.net/biology-qualifiers/',
              'bqmodel': 'http://biomodels.net/model-qualifiers/',
              'vCard': 'http://www.w3.org/2001/vcard-rdf/3.0#',
              'dc': 'http://purl.org/dc/elements/1.1/'}


def register_all_namespaces(fname):
    namespaces = dict([node for _, node in
                       ET.iterparse(fname, events=['start-ns'])])
    for ns in namespaces:
        ET.register_namespace(ns, namespaces[ns])


def add_groundings(et):
    species = et.findall('sbml:model/sbml:listOfSpecies/sbml:species',
                         namespaces=namespaces)
    for species in species:
        name = species.attrib.get('name')
        print(name)
        class_tag = species.find(
            'sbml:annotation/celldesigner:extension/'
            'celldesigner:speciesIdentity/celldesigner:class',
            namespaces=namespaces)
        if class_tag is not None:
            cls = class_tag.text
            print(cls)
        print(get_existing_grounding(species))
        matches = gilda.ground(name)
        if matches:
            print(name, matches[0].term.db, matches[0].term.id,
                  matches[0].term.entry_name)
            add_grounding_element(species, matches[0].term.db,
                                  matches[0].term.id)
        print('---')
    return et


def get_existing_grounding(species):
    grounding_elements = \
        species.findall('sbml:annotation/rdf:RDF/rdf:Description/'
                        'bqbiol:isDescribedBy/rdf:Bag/rdf:li',
                        namespaces=namespaces)
    groundings = defaultdict(list)
    for element in grounding_elements:
        urn = element.attrib['{http://www.w3.org/1999/02/22-rdf-syntax-ns#}'
                             'resource']
        match = re.match(r'urn:miriam:([^:]+):(.+)', urn)
        if not match:
            print('Unmatched urn: %s' % urn)
            continue
        else:
            db_ns, db_id = match.groups()
            groundings[db_ns].append(db_id)
    return groundings


def add_grounding_element(species, db_ns, db_id):
    identifiers_ns = identifiers.get_identifiers_ns(db_ns)
    grounding_str = 'urn:miriam:%s:%s' % (identifiers_ns, db_id)

    tag_sequence = [
        'sbml:annotation',
        'rdf:RDF',
        'rdf:Description',
        'bqmodel:is',
        'rdf:Bag',
    ]
    root = species
    for tag in tag_sequence:
        element = root.find(tag, namespaces=namespaces)
        if element is not None:
            root = element
        else:
            new_element = ET.Element(tag)
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
    xml_str = xml_str.replace(b' />', b'/>')
    xml_str = xml_str.replace(b'<html>',
                              b'<html xmlns="http://www.w3.org/1999/xhtml">')
    xml_str = xml_str.replace(b'<rdf:RDF>', rdf_str)
    with open(fname, 'wb') as fh:
        fh.write(xml_str)


if __name__ == '__main__':
    base_path = sys.argv[1]
    stable_xmls = list(Path(base_path).rglob('*_stable.xml'))
    for stable_xml in stable_xmls[:1]:
        print('Grounding %s' % stable_xml)
        register_all_namespaces(stable_xml)
        et = ET.parse(stable_xml)
        et = add_groundings(et)
        out_fname = stable_xml.as_posix()[:-4] + '_grounded.xml'
        dump(et, out_fname)
