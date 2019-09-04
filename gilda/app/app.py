from flask import Flask, abort, Response, request, jsonify
from gilda.grounder import Grounder
from gilda.resources import get_grounding_terms

app = Flask(__name__)

grounder = Grounder(get_grounding_terms())


@app.route('/ground', methods=['POST'])
def ground():
    if request.json is None:
        abort(Response('Missing application/json header.', 415))
    # Get input parameters
    text = request.json.get('text')
    context = request.json.get('context')
    scored_matches = grounder.ground(text)
    if context:
        scored_matches = grounder.disambiguate(text, scored_matches, context)
    res = []
    for scored_match in sorted(scored_matches, key=lambda x: x.score,
                               reverse=True):
        res.append(scored_match.to_json())
    return jsonify(res)
