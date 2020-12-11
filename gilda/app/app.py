from flask import Flask, Response, abort, jsonify, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField, \
    SelectMultipleField
from wtforms.validators import DataRequired

from gilda.api import *
from gilda import __version__ as version
from gilda.resources import popular_organisms

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False
Bootstrap(app)


class GroundForm(FlaskForm):
    text = StringField('Text', validators=[DataRequired()])
    context = TextAreaField('Context')
    organisms = SelectMultipleField('Organisms',
                                    choices=[(org, org)
                                             for org in popular_organisms],
                                    id='organism-select')
    submit = SubmitField('Submit')

    def get_matches(self):
        return ground(self.text.data, context=self.context.data,
                      organisms=self.organisms.data)


@app.route('/', methods=['GET', 'POST'])
def info():
    form = GroundForm()
    if form.validate_on_submit():
        matches = form.get_matches()
        return render_template('matches.html', matches=matches, form=form,
                               version=version)
    return render_template('home.html', form=form, version=version)


@app.route('/ground', methods=['POST'])
def ground_endpoint():
    if request.json is None:
        abort(Response('Missing application/json header.', 415))
    # Get input parameters
    text = request.json.get('text')
    context = request.json.get('context')
    organisms = request.json.get('organisms')
    scored_matches = ground(text, context=context, organisms=organisms)
    res = [sm.to_json() for sm in scored_matches]
    return jsonify(res)


@app.route('/get_names', methods=['POST'])
def get_names_endpoint():
    if request.json is None:
        abort(Response('Missing application/json header.', 415))
    # Get input parameters
    kwargs = {key: request.json.get(key) for key in {'db', 'id', 'status',
                                                     'source'}}
    names = get_names(**kwargs)
    return jsonify(names)


@app.route('/models', methods=['GET', 'POST'])
def models():
    return jsonify(get_models())
