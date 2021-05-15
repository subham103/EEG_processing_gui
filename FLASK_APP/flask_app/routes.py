"""Routing."""
from flask import current_app as app
from flask import redirect, render_template, url_for
import sys
import json
from flask.globals import request

from matplotlib.pyplot import title
from .forms import VisualizeForm, FeatureForm
from .PHN_300_flask import till_display, till_classification


@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    """Landing page."""
    form = VisualizeForm()
    if form.validate_on_submit():
        print(form.data, file=sys.stderr)
        till_display(form.data)
        with open('visualize_dump.json', 'w') as f:
            json.dump(form.data, f)
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
        visualize_form_data = json.loads(open("visualize_dump.json","r").read())
        with open('visualize_dump.json', 'w') as f:
            json.dump({}, f)
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
    return render_template(
        "success.jinja2",
        template="success-template"
    )
