"""This script grounds entries in the MedMentions data set available at
https://github.com/chanzuckerberg/MedMentions. This can serve both as a
benchmark and as a bsis for creating mappings between the namespaces
Gilda grounds to and UMLS, which is used for MedMentions groundings.

.. code-block:: bibtex

    @article{Mohan2019,
        archivePrefix = {arXiv},
        arxivId = {1902.09476},
        author = {Mohan, Sunil and Li, Donghui},
        eprint = {1902.09476},
        month = {feb},
        title = {{MedMentions: A Large Biomedical Corpus Annotated with UMLS Concepts}},
        url = {http://arxiv.org/abs/1902.09476},
        year = {2019}
    }
"""

import csv
import json

import click
import gilda
import pyobo
import pystow
from more_click import verbose_option
from pubtator_loader import from_gz
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

URL = "https://github.com/chanzuckerberg/MedMentions/raw/master/full/data/corpus_pubtator.txt.gz"
MODULE = pystow.module("gilda", "medmentions")
CORPUS_PATH = MODULE.join(name="corpus.json")
MATCHING_PATH = pystow.join(
    "gilda", "medmentions", gilda.__version__, name="matching.tsv"
)

#: Subset of types from the Semantic Type Ontology used in annotation
TYPES = {
    "T116": "Amino Acid, Peptide, or Protein",
    "T123": "Biologically Active Substance",
    "T047": "Disease or Syndrome",
    "T101": "Patient or Disabled Group",
    "T079": "Temporal Concept",
    "T169": "Functional Concept",
    "T033": "Finding",
    "T081": "Quantitative Concept",
    "T063": "Molecular Biology Research Technique",
    "T052": "Activity",
    "T062": "Research Activity",
    "T032": "Organism Attribute",
    "T098": "Population Group",
    "T100": "Age Group",
    "T073": "Manufactured Object",
    "T093": "Health Care Related Organization",
    "T026": "Cell Component",
    "T028": "Gene or Genome",
    "T007": "Bacterium",
    "T045": "Genetic Function",
    "T046": "Pathologic Function",
    "T131": "Hazardous or Poisonous Substance",
    "T043": "Cell Function",
    "T025": "Cell",
    "T067": "Phenomenon or Process",
    "T069": "Environmental Effect of Humans",
    "T167": "Substance",
    "T037": "Injury or Poisoning",
    "T001": "Organism",
    "T031": "Body Substance",
    "T080": "Qualitative Concept",
    "T121": "Pharmacologic Substance",
    "T196": "Element, Ion, or Isotope",
    "T044": "Molecular Function",
    "T126": "Enzyme",
    "T077": "Conceptual Entity",
    "T082": "Spatial Concept",
    "T103": "Chemical",
    "T078": "Idea or Concept",
    "T039": "Physiologic Function",
    "T109": "Organic Chemical",
    "T122": "Biomedical or Dental Material",
    "T074": "Medical Device",
    "T058": "Health Care Activity",
    "T070": "Natural Phenomenon or Process",
    "T057": "Occupational Activity",
    "T023": "Body Part, Organ, or Organ Component",
    "T059": "Laboratory Procedure",
    "T042": "Organ or Tissue Function",
    "T060": "Diagnostic Procedure",
    "T184": "Sign or Symptom",
    "T024": "Tissue",
    "T029": "Body Location or Region",
    "T053": "Behavior",
    "T061": "Therapeutic or Preventive Procedure",
    "T083": "Geographic Area",
    "T055": "Individual Behavior",
    "T185": "Classification",
    "UnknownType": "UnknownType",
    "T054": "Social Behavior",
    "T170": "Intellectual Product",
    "T097": "Professional or Occupational Group",
    "T041": "Mental Process",
    "T095": "Self-help or Relief Organization",
    "T099": "Family Group",
    "T048": "Mental or Behavioral Dysfunction",
    "T071": "Entity",
    "T204": "Eukaryote",
    "T168": "Food",
    "T019": "Congenital Abnormality",
    "T201": "Clinical Attribute",
    "T040": "Organism Function",
    "T191": "Neoplastic Process",
    "T049": "Cell or Molecular Dysfunction",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T002": "Plant",
    "T090": "Occupation or Discipline",
    "T096": "Group",
    "T075": "Research Device",
    "T034": "Laboratory or Test Result",
    "T197": "Inorganic Chemical",
    "T195": "Antibiotic",
    "T091": "Biomedical Occupation or Discipline",
    "T008": "Animal",
    "T102": "Group Attribute",
    "T012": "Bird",
    "T011": "Amphibian",
    "T200": "Clinical Drug",
    "T125": "Hormone",
    "T129": "Immunologic Factor",
    "T127": "Vitamin",
    "T056": "Daily or Recreational Activity",
    "T087": "Amino Acid Sequence",
    "T086": "Nucleotide Sequence",
    "T005": "Virus",
    "T065": "Educational Activity",
    "T192": "Receptor",
    "T022": "Body System",
    "T015": "Mammal",
    "T038": "Biologic Function",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
    "T030": "Body Space or Junction",
    "T017": "Anatomical Structure",
    "T068": "Human-caused Phenomenon or Process",
    "T016": "Human",
    "T050": "Experimental Model of Disease",
    "T089": "Regulation or Law",
    "T085": "Molecular Sequence",
    "T104": "Chemical Viewed Structurally",
    "T064": "Governmental or Regulatory Activity",
    "T190": "Anatomical Abnormality",
    "T092": "Organization",
    "T051": "Event",
    "T004": "Fungus",
    "T171": "Language",
    "T014": "Reptile",
    "T120": "Chemical Viewed Functionally",
    "T020": "Acquired Abnormality",
    "T013": "Fish",
    "T194": "Archaeon",
    "T072": "Physical Object",
    "T018": "Embryonic Structure",
    "T066": "Machine Activity",
    "T010": "Vertebrate",
    "T094": "Professional Society",
    "T021": "Fully Formed Anatomical Structure",
    "T203": "Drug Delivery Device",
}


def get_corpus():
    """Get the MedMentions corpus.

    :return: A list of dictionaries of the following form:

    .. code-block::

        [
          {
            "id": 25763772,
            "title_text": "DCTN4 as a modifier of chronic Pseudomonas aeruginosa infection in cystic fibrosis",
            "abstract_text": "...",
            "entities": [
              {
                "document_id": 25763772,
                "start_index": 0,
                "end_index": 5,
                "text_segment": "DCTN4",
                "semantic_type_id": "T116,T123",
                "entity_id": "C4308010"
              },
              ...
            ]
          },
          ...
        ]
    """
    if not CORPUS_PATH.is_file():
        path = MODULE.ensure(url=URL)
        corpus = from_gz(path)
        with CORPUS_PATH.open("w") as file:
            json.dump(corpus, file, indent=2, default=lambda o: o.__dict__)

    # Right now I'd rather not engage with the strange object structure, so
    # serializing and deserializing gets us JSON we can work with.
    with CORPUS_PATH.open() as file:
        return json.load(file)


def iterate_corpus():
    corpus = get_corpus()
    click.echo(f"MedMentions has {len(corpus)} entries")
    for document in tqdm(corpus, unit="document", desc="MedMentions"):
        document_id = document["id"]
        abstract = document["abstract_text"]
        for entity_idx, entity in enumerate(document["entities"]):
            umls_id = entity["entity_id"]
            text = entity["text_segment"]
            start, end = entity["start_index"], entity["end_index"]
            types = set(entity["semantic_type_id"].split(","))
            yield document_id, abstract, umls_id, text, start, end, types


HEADER = [
    "document",
    "start_idx",
    "end_idx",
    "text",
    "umls_id",
    "umls_name",
    "gilda_prefix",
    "gilda_identifier",
    "gilda_name",
    "gilda_score",
]


@click.command()
@verbose_option
def main():
    corpus = get_corpus()
    click.echo(f"There are {len(corpus)} entries")
    rows = []
    for document_id, abstract, umls_id, text, start, end, types in iterate_corpus():
        with logging_redirect_tqdm():
            matches = gilda.ground(text, context=abstract)
        rows.extend(
            (
                document_id,
                start,
                end,
                text,
                umls_id,
                pyobo.get_name("umls", umls_id),
                match.term.db,
                match.term.id,
                match.term.entry_name,
                match.score,
            )
            for match in matches
        )
    with MATCHING_PATH.open("w") as file:
        writer = csv.writer(file, delimiter="\t")
        writer.writerow(HEADER)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
