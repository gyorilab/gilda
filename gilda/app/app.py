from flask import Flask, Response, abort, jsonify, render_template, request, \
    make_response
from flask_bootstrap import Bootstrap
from flask_restx import Api, Resource, fields
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, \
    SelectMultipleField
from wtforms.validators import DataRequired

from gilda.api import *
from gilda import __version__ as version
from gilda.resources import popular_organisms

app = Flask(__name__)
app.config['RESTX_MASK_SWAGGER'] = True
app.config['WTF_CSRF_ENABLED'] = False
Bootstrap(app)


class GroundForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    context = TextAreaField('Context (optional)')
    organisms = SelectMultipleField('Species priority (optional)',
                                    choices=[(org, org)
                                             for org in popular_organisms],
                                    id='organism-select')
    submit = SubmitField('Submit')

    def get_matches(self):
        return ground(self.text.data, context=self.context.data,
                      organisms=self.organisms.data)


@app.route('/')
def home():
    form = GroundForm()
    if form.validate_on_submit():
        matches = form.get_matches()
        return render_template('matches.html', matches=matches, form=form,
                               version=version)
    return render_template('home.html', form=form, version=version)


# NOTE: the Flask REST-X API has to be declared here, below the home endpoint
# otherwise it reserves the / base path.

api = Api(app,
          title="Gilda",
          description="A service for grounding entity strings",
          version=version,
          license="Code available under the BSD 2-Clause License",
          contact="benjamin_gyori@hms.harvard.edu",
          doc='/apidoc',
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
    {'norm_text' : fields.String(
        description='The normalized text corresponding to the text entry, '
                    'used for lookups.',
        example='egf receptor'),
     'text' : fields.String(
         description='The text entry that was matches.',
         example='EGF receptor'
     ),
     'db' : fields.String(
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
         example='assertion'
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
     )}
)

scored_match_model = api.model(
    "Scored match",
    {'term': fields.Nested(term_model,
                           description='The term that was matched'),
     'url': fields.String(
         description='Identifiers.org URL for the matched term.',
         example='https://identifiers.org/hgnc:3236'
     ),
     'score': fields.Float(
         description='The score assigned to the matched term.',
         example=0.9845
     ),
     'match': fields.Nested(
         description='Additional metadata about the nature of the match.'
     )}
)

scored_match_list_model = api.model(
    "Scored match list", {'': fields.List(scored_match_model)}
)


@base_ns.route('/ground', methods=['POST'])
class Ground(Resource):
    @base_ns.response(200, "Grounding results")
    @base_ns.expect(grounding_input_model)
    @base_ns.marshal_with(scored_match_list_model)
    def post(self):
        if request.json is None:
            abort(Response('Missing application/json header.', 415))
        # Get input parameters
        text = request.json.get('text')
        context = request.json.get('context')
        organisms = request.json.get('organisms')
        scored_matches = ground(text, context=context, organisms=organisms)
        res = [sm.to_json() for sm in scored_matches]
        return jsonify(res)


#@app.route('/get_names', methods=['POST'])
def get_names_endpoint():
    """Return all known entity texts (names, synonyms, etc.) for a grounding.

    This endpoint can be used as a reverse lookup to find out what entity texts
    are known for a given grounded entity

    ---
    parameters:
    - name: db
      in: body
      type: string
      description: "Capitalized name of the database for the grounding, e.g.
        HGNC."
      required: true
      example: HGNC
    - name: id
      in: body
      type: string
      description: "Identifier within the given database"
      required: true
      example: 6872
    - name: status
      in: body
      type: string
      description: "If provided, only entity texts of the given status are
        returned (e.g., assertion, name, synonym, previous)."
      required: false
      example: synonym
    - name: source
      in: body
      type: string
      description: "If provided, only entity texts collected from the given
        source are returned. This is useful if terms grounded to IDs in a given
        database are collected from multiple different sources."
      required: false
      example: hgnc

    responses:
      200:
        description: A list of entity texts for the given grounding.
    """
    if request.json is None:
        abort(Response('Missing application/json header.', 415))
    # Get input parameters
    kwargs = {key: request.json.get(key) for key in {'db', 'id', 'status',
                                                     'source'}}
    names = get_names(**kwargs)
    return jsonify(names)


#@app.route('/models', methods=['GET', 'POST'])
def models():
    """Return a list of entity texts with Gilda disambiguation models.

    Gilda makes available more than one thousand disambiguation models
    between synonyms shared by multiple genes. This endpoint returns
    the list of entity texts for which such a model is available.
    """
    return jsonify(get_models())
