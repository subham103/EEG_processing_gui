"""Routing."""
from flask import current_app as app
from flask import redirect, render_template, url_for
import sys
import json
from flask.globals import request, session

from matplotlib.pyplot import title
from .forms import VisualizeForm, FeatureForm
from .PHN_300_flask import till_display, till_classification


# A welcome message to test the server
@app.route('/')
def index():
    return "<h1>Welcome to our EEG Project !!</h1>"

@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    """Landing page."""
    form = VisualizeForm()
    if form.validate_on_submit():
        print(form.data, file=sys.stderr)
        till_display(form.data)
        # with open('visualize_dump.json', 'w') as f:
        #     json.dump(form.data, f)
        session["visualize_data"] = form.data
        return redirect(url_for("feature"))
    return render_template(
        "index.jinja2",
        form=form,
        template="form-template",
        title="Input I"
    )

@app.route("/feature", methods=["GET", "POST"])
def feature():
    """Feature Selection Page."""
    form = FeatureForm()
    # print(request.args.get('obj'), file=sys.stderr)
    if form.validate_on_submit():
        # feature_form_data = form.data
        # visualize_form_data = json.loads(open("visualize_dump.json","r").read())
        # with open('visualize_dump.json', 'w') as f:
        #     json.dump({}, f)
        visualize_form_data = session["visualize_data"]
        till_classification(visualize_form_data, form.data)
        return redirect(url_for("success"))
    return render_template(
        "feature_selection.jinja2",
        form=form,
        template="form-template",
        title="Input II"
    )

@app.route("/success", methods=["GET", "POST"])
def success():
    """Generic success page upon form submission."""
    table=json.loads(open("table.json","r").read())
    return render_template(
        "success.jinja2",
        template="success-template",
        table=table,
    )
