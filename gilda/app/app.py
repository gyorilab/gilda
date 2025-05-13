import logging
from textwrap import dedent
from typing import Optional

from flask import Flask, abort, jsonify, redirect, request
from flask_restx import Api, Resource, fields

from gilda import __version__ as version
from gilda.grounder import GrounderInput, Grounder
from gilda.app.proxies import grounder
from gilda.ner import annotate

# NOTE: the Flask REST-X API has to be declared here, below the home endpoint
# otherwise it reserves the / base path.

logger = logging.getLogger(__name__)

api = Api(title="Gilda",
          description="A service for grounding entity strings",
          version=version,
          license="Code available under the BSD 2-Clause License",
          contact="INDRA labs",
          contact_email="indra.sysbio@gmail.com",
          contact_url="https://indralab.github.io",
          doc='/apidocs',
          )

base_ns = api.namespace('Gilda API', 'Gilda API', path='/')

grounding_input_model = api.model(
    "GroundingInput",
    {'text': fields.String(example='EGF-receptor',
                           required=True,
                           description='The entity string to be grounded.'),
     'context': fields.String(example='The EGFR-receptor binds EGF.',
                              required=False,
                              description='Surrounding text as context to aid'
                                          ' disambiguation when applicable.'),
     'organisms': fields.List(fields.String, example=['9606'],
                              description='An optional list of taxonomy '
                                          'species IDs defining a priority list'
                                          ' in case an entity string can be '
                                          'resolved to multiple'
                                          'species-specific genes/proteins.',
                              required=False)}
)

term_model = api.model(
    "Term",
    {'norm_text': fields.String(
        description='The normalized text corresponding to the text entry, '
                    'used for lookups.',
        example='egf receptor'),
     'text': fields.String(
         description='The text entry that was matched.',
         example='EGF receptor'
     ),
     'db': fields.String(
         description='The database / namespace corresponding to the '
                     'grounded term.',
         example='HGNC'
     ),
     'id': fields.String(
         description='The identifier of the grounded term within the '
                     'database / namespace.',
         example='3236'
     ),
     'entry_name': fields.String(
         description='The standardized name corresponding to the grounded '
                     'term.',
         example='EGFR'
     ),
     'status': fields.String(
         description='The relationship of the text entry to the grounded '
                     'term, e.g., synonym.',
         example='curated'
     ),
     'source': fields.String(
         description='The source from which the term was obtained.',
         example='famplex'
     ),
     'organism': fields.String(
         description='If the term is a gene/protein, this field provides '
                     'the taxonomy identifier of the species to which '
                     'it belongs.',
         example='9606'
     ),
     'source_db': fields.String(
         description='In some cases the term\'s db/id was mapped from another '
                     'db/id pair given in the original source. If this is the '
                     'case, this field provides the original source db.'),
     'source_id': fields.String(
         description='In some cases the term\'s db/id was mapped from another '
                     'db/id pair given in the original source. If this is the '
                     'case, this field provides the original source ID.')
    }
)

scored_match_model = api.model(
    "ScoredMatch",
    {'term': fields.Nested(term_model, description='The term that was matched'),
     'url': fields.String(
         description='bioregistry.io URL for the matched term.',
         example='https://bioregistry.io/hgnc:3236'
     ),
     'score': fields.Float(
         description='The score assigned to the matched term.',
         example=0.9845
     ),
     'match': fields.Nested(api.model('Match', {}),
         description='Additional metadata about the nature of the match.'
     ),
     'subsumed_terms': fields.List(fields.Nested(term_model),
         description='In some cases multiple terms with the same db/id '
                     'matched the input string, potentially with different '
                     'scores, and only the first one is exposed in the '
                     'scored match\'s term attribute (see above). This field '
                     'provides additional terms with the same db/id that '
                     'matched the input for additional traceability.')
     }
)

get_names_input_model = api.model(
    "GetNamesInput",
    {'db': fields.String(
         description="Capitalized name of the database for the grounding, "
                     "e.g. HGNC.",
         required=True,
         example='HGNC'),
     'id': fields.String(
         description="Identifier within the given database",
         required=True,
         example='3236'
     ),
     'status': fields.String(
         description="If provided, only entity texts of the given status are "
                     "returned (e.g., curated, name, synonym, former_name).",
         required=False,
         enum=['curated', 'name', 'synonym', 'former_name'],
         example='synonym'
     ),
     'source': fields.String(
         description="If provided, only entity texts collected from the given "
                     "source are returned.This is useful if terms grounded to "
                     "IDs in a given database are collected from multiple "
                     "different sources.",
         required=False,
         example='uniprot'
    )
    }
)

ner_result_model = api.model('NERResult', {
    'text': fields.String(description='Matched text'),
    'start': fields.Integer(description='Start index of the match'),
    'end': fields.Integer(description='End index of the match'),
    'matches': fields.List(fields.Nested(scored_match_model))
})

ner_input_model = api.model('NERInput', {
    'text': fields.String(required=True, description='Text on which to perform'
                                                     ' NER',
                          example='The EGF receptor binds EGF which is an '
                                  'interaction important in cancer.'),
    'organisms': fields.List(fields.String, example=['9606'],
                             description='An optional list of taxonomy '
                                         'species IDs defining a priority list'
                                         ' in case an entity string can be '
                                         'resolved to multiple'
                                         'species-specific genes/proteins.',
                             required=False),
    'namespaces': fields.List(fields.String,
                              description='A list of namespaces to pass to '
                                          'the grounder to restrict the '
                                          'matches to. By default, '
                                          'no restriction is applied',
                              example=['HGNC', 'MESH'],
                              required=False),
    'context_text': fields.String(required=False, description='Additional '
                                                              'context for '
                                                              'disambiguation',
                                  example='The EGF receptor binds EGF which is an interaction'
                                          'important in cancer.'),
})

names_model = fields.List(
    fields.String,
    example=['EGF receptor', 'EGFR', 'ERBB1', 'Proto-oncogene c-ErbB-1'])

models_model = fields.List(
    fields.String,
    example=['A4', 'ABC1', 'p180'])


@base_ns.route('/ground', methods=['POST'])
class Ground(Resource):
    @base_ns.response(200, "Grounding results", [scored_match_model])
    @base_ns.expect(grounding_input_model)
    def post(self):
        """Perform grounding on a given entity text.

        Returns a list of scored grounding matches for the given entity text.
        The returned value is a list with each entry being a scored match.
        Each scored match contains a term which was matched, and each term
        contains a db and id constituting a grounding. An empty list
        return value means that no grounding matches were found for the input.
        """
        if request.json is None:
            abort(415, 'Missing application/json header.')
        # Get input parameters
        text = request.json.get('text')
        context = request.json.get('context')
        organisms = request.json.get('organisms')
        scored_matches = grounder.ground(text, context=context,
                                         organisms=organisms)
        res = [sm.to_json() for sm in scored_matches]
        return jsonify(res)


@base_ns.route('/ground_multi', methods=['POST'])
class GroundMulti(Resource):
    @base_ns.response(200, "Grounding results", [[scored_match_model]])
    @base_ns.expect([grounding_input_model])
    def post(self):
        """Return a list of grounding matches for a list of inputs.

        This endpoint is useful for batch processing of inputs. The input
        is a list with each entry containing a text key as well as optional
        context and organism keys. The returned value is a list corresponding
        to each input in order, each entry of the returned list being a list
        of scored matches.
        """
        if request.json is None:
            abort(415, 'Missing application/json header.')
        # Get input parameters
        all_matches = []
        for input in request.json:
            text = input.get('text')
            context = input.get('context')
            organisms = input.get('organisms')
            scored_matches = grounder.ground(text, context=context,
                                             organisms=organisms)
            all_matches.append([sm.to_json() for sm in scored_matches])
        return jsonify(all_matches)


@base_ns.route('/names', methods=['POST'])
@base_ns.route('/get_names', methods=['POST'])
class GetNames(Resource):
    @base_ns.response(200, "Get names result", names_model)
    @base_ns.expect(get_names_input_model)
    def post(self):
        """Return all known entity texts for a grounding.

        This endpoint can be used as a reverse lookup to find out what entity
        texts (names, synonyms, etc.) are known for a given grounded entity.
        """
        if request.json is None:
            abort(415, 'Missing application/json header.')
        # Get input parameters
        kwargs = {key: request.json.get(key) for key in {'db', 'id', 'status',
                                                         'source'}}
        names = grounder.get_names(**kwargs)
        return jsonify(names)


@base_ns.route('/models', methods=['GET', 'POST'])
class GetModels(Resource):
    @base_ns.response(200, "Get models result", models_model)
    def post(self):
        """Return a list of texts with Gilda disambiguation models.

        Gilda makes available more than one thousand disambiguation models
        between synonyms shared by multiple genes. This endpoint returns
        the list of entity texts for which such a model is available.
        """
        return jsonify(grounder.get_models())

    def get(self):
        """Return a list of texts with Gilda disambiguation models.

        Gilda makes available more than one thousand disambiguation models
        between synonyms shared by multiple genes. This endpoint returns
        the list of entity texts for which such a model is available.
        """
        return jsonify(grounder.get_models())


@base_ns.route('/annotate', methods=['POST'])
class Annotate(Resource):
    @base_ns.response(200, "NER results", [ner_result_model])
    @base_ns.expect(ner_input_model)
    def post(self):
        """Perform named entity recognition on the given text.

        This endpoint can be used to perform named entity recognition (NER)
        using Gilda.
        """
        from gilda.ner import annotate

        if request.json is None:
            abort(415, 'Missing application/json header.')

        text = request.json.get('text')
        context_text = request.json.get('context_text')
        organisms = request.json.get('organisms')
        namespaces = request.json.get('namespaces')

        results = annotate(text, organisms=organisms if organisms else None,
                           namespaces=namespaces if namespaces else None,
                           context_text=context_text)
        return jsonify([annotation.to_json() for annotation in results])


def get_app(terms: Optional[GrounderInput] = None, *, ui: bool = True) -> Flask:
    app = Flask(__name__)
    app.config['RESTX_MASK_SWAGGER'] = False
    app.config['WTF_CSRF_ENABLED'] = False
    app.config['SWAGGER_UI_DOC_EXPANSION'] = 'list'
    app.config["grounder"] = Grounder(terms=terms)

    if not ui:
        _mount_home_redirect(app)
    else:
        try:
            import importlib.metadata as importlib_metadata
        except ImportError:
            import importlib_metadata

        try:
            from flask_bootstrap import Bootstrap

            bootstrap_version = importlib_metadata.version("flask_bootstrap")
            if "3.3.7.1" != bootstrap_version:
                raise ImportError(
                    dedent(
                        """\
                    The wrong flask-bootstrap is installed, therefore the UI
                    can not be enabled. Please run the following commands in
                    the shell:

                      pip uninstall flask-bootstrap bootstrap-flask
                      pip install flask-bootstrap
                    """
                    )
                )

            from gilda.app.ui import ui_blueprint
        except ImportError as e:
            logger.error('Could not import UI blueprint: %s', e)
            _mount_home_redirect(app)
        else:
            Bootstrap(app)
            app.register_blueprint(ui_blueprint, url_prefix="/")

    # has to be put after defining the UI blueprint otherwise it reserves "/"
    api.init_app(app)
    return app


def _mount_home_redirect(app):
    @app.route("/")
    def home_redirect():
        """Redirect the home url to the API documentation."""
        return redirect("/apidocs")
