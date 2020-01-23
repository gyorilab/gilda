from flask import Flask, abort, Response, request, jsonify
from gilda.api import *

app = Flask(__name__)


@app.route('/', methods=['GET'])
def info():
    res = ('This is the Gilda grounding service. Please POST to the /ground '
           'endpoint for grounding and see the documentation at '
           'http://github.com/indralab/gilda for more information.')
    return res


@app.route('/ground', methods=['POST'])
def ground_endpoint():
    if request.json is None:
        abort(Response('Missing application/json header.', 415))
    # Get input parameters
    text = request.json.get('text')
    context = request.json.get('context')
    scored_matches = ground(text, context)
    res = [sm.to_json() for sm in scored_matches]
    return jsonify(res)


@app.route('/models', methods=['GET', 'POST'])
def models():
    return jsonify(get_models())
