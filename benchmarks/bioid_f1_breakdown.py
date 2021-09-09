import os
import json

from bioid_evaluation import BioIDBenchmarker


HERE = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(HERE, 'data')
try:
    with open(os.path.join(data_path, 'equivalences.json')) as f:
        equivalences = json.load(f)
except FileNotFoundError:
    equivalences = {}

benchmarker = BioIDBenchmarker(equivalences=equivalences)
benchmarker.ground_entities_with_gilda(context=True)

df = benchmarker.processed_data
df['top_grounding_no_context'] = df.groundings_no_context.\
    apply(lambda x: x[0][0] if x else None)

fp = \
    df[
        (~df.top_correct_no_context) &
        (
            df.top_grounding_no_context.apply(
                lambda x: x is not None and x.startswith('HGNC'))
        )
    ]

tp = \
    df[
        (df.top_correct_no_context) &
        (
            df.top_grounding_no_context.apply(
                lambda x: x is not None and x.startswith('HGNC')
            )
        )
    ]

pos = df[
    df.obj_synonyms.apply(lambda x: any([y.startswith('HGNC') for y in x]))
]


pr = len(tp) / (len(tp) + len(fp))
rc = len(tp) / len(pos)
f1 = 2/(1/pr + 1/rc)


df['top_grounding'] = df.groundings.\
    apply(lambda x: x[0][0] if x else None)

fp_disamb = \
    df[
        (~df.top_correct) &
        (
            df.top_grounding.apply(
                lambda x: x is not None and x.startswith('HGNC'))
        )
    ]

tp_disamb = \
    df[
        (df.top_correct) &
        (
            df.top_grounding.apply(
                lambda x: x is not None and x.startswith('HGNC')
            )
        )
    ]

pos_disamb = df[
    df.obj_synonyms.apply(lambda x: any([y.startswith('HGNC') for y in x]))
]


pr_disamb = len(tp_disamb) / (len(tp_disamb) + len(fp_disamb))
rc_disamb = len(tp_disamb) / len(pos_disamb)
f1_disamb = 2/(1/pr_disamb + 1/rc_disamb)
