from textwrap import dedent

from flask import Blueprint, render_template, request
from flask_wtf import FlaskForm
from wtforms import SelectMultipleField, StringField, SubmitField, TextAreaField
from wtforms.validators import DataRequired

from gilda.app.proxies import grounder
from gilda import __version__ as version
from gilda.resources import organism_labels, popular_organisms

__all__ = [
    "ui_blueprint",
]

ORGANISMS_FIELD = SelectMultipleField(
    "Species priority (optional)",
    choices=[(org, organism_labels[org]) for org in popular_organisms],
    id="organism-select",
    description=dedent(
        """\
        Optionally select one or more taxonomy
        species IDs to define a species priority list.  Click
        <a type="button" href="#" data-toggle="modal" data-target="#species-modal">
        here <i class="far fa-question-circle">
        </i></a> for more details.
    """
    ),
)


class GroundForm(FlaskForm):
    text = StringField(
        "Text",
        validators=[DataRequired()],
        description="Input the entity text (e.g., <code>k-ras</code>) to ground.",
    )
    context = TextAreaField(
        "Context (optional)",
        description=dedent(
            """\
            Optionally provide additional text context to help disambiguation. Click
            <a type="button" href="#" data-toggle="modal" data-target="#context-modal">
            here <i class="far fa-question-circle">
            </i></a> for more details.
        """
        ),
    )
    organisms = ORGANISMS_FIELD
    submit = SubmitField("Submit")

    def get_matches(self):
        return grounder.ground(
            self.text.data, context=self.context.data, organisms=self.organisms.data
        )


class NERForm(FlaskForm):
    text = TextAreaField(
        "Text",
        validators=[DataRequired()],
        description=dedent(
            """\
            Text from which to identify and ground named entities.
        """
        ),
    )
    organisms = ORGANISMS_FIELD
    submit = SubmitField("Submit")

    def get_annotations(self):
        from gilda.ner import annotate

        return annotate(
            self.text.data, grounder=grounder, organisms=self.organisms.data
        )


ui_blueprint = Blueprint("ui", __name__, url_prefix="/")


@ui_blueprint.route("/", methods=["GET", "POST"])
def home():
    text = request.args.get("text")
    if text is not None:
        context = request.args.get("context")
        organisms = request.args.getlist("organisms")
        matches = grounder.ground(text, context=context, organisms=organisms)
        return render_template(
            "matches.html", matches=matches, version=version, text=text, context=context
        )

    form = GroundForm()
    if form.validate_on_submit():
        matches = form.get_matches()
        return render_template(
            "matches.html",
            matches=matches,
            version=version,
            text=form.text.data,
            context=form.context.data,
            # Add a new form that doesn't auto-populate
            form=GroundForm(formdata=None),
        )
    return render_template("home.html", form=form, version=version)


@ui_blueprint.route("/ner", methods=["GET", "POST"])
def view_ner():
    form = NERForm()
    if form.validate_on_submit():
        annotations = form.get_annotations()
        return render_template(
            "ner_matches.html",
            annotations=annotations,
            version=version,
            text=form.text.data,
            # Add a new form that doesn't auto-populate
            form=NERForm(formdata=None),
        )
    return render_template("ner_home.html", form=form, version=version)
