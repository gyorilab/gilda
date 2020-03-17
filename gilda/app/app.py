import os

from flask import Flask, Response, abort, jsonify, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, TextAreaField

from gilda.api import *

app = Flask(__name__)
app.secret_key = os.urandom(8)
Bootstrap(app)


class GroundForm(FlaskForm):
    text = StringField('Text')
    context = TextAreaField('Context')
    submit = SubmitField('Submit')


@app.route('/', methods=['GET'])
def info():
    return render_template('home.html', form=GroundForm())


@app.route('/ground', methods=['POST'])
def ground_endpoint():
    content_type = request.headers["Content-Type"]
    if "application/json" in content_type:
        # Get input parameters
        text = request.json.get('text')
        context = request.json.get('context')
    elif 'application/x-www-form-urlencoded' in content_type:
        # Get input parameters from form
        form = GroundForm()
        text = form.text.data
        context = form.context.data
    else:
        return abort(Response('Missing application/json header.'
                              ' Got {}'.format(content_type), 415))

    scored_matches = ground(text, context)
    res = [sm.to_json() for sm in scored_matches]
    return jsonify(res)


@app.route('/models', methods=['GET', 'POST'])
def models():
    return jsonify(get_models())
