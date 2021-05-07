"""Routing."""
from flask import current_app as app
from flask import redirect, render_template, url_for
import sys
from .forms import VisualizeForm
from .aa import till_display

@app.route("/visualize", methods=["GET", "POST"])
def visualize():
    """Landing page."""
    form = VisualizeForm()
    # print(form.submit, file=sys.stderr)
    if form.validate_on_submit():
        # print(form.data, file=sys.stderr)
        vall = till_display(form.data)
        print(vall, file=sys.stderr)
        return redirect(url_for("success"))
    return render_template(
        "index.jinja2",
        form=form,
        template="form-template",
        title="Input I"
    )

@app.route("/success", methods=["GET", "POST"])
def success():
    """Generic success page upon form submission."""
    return render_template(
        "success.jinja2",
        template="success-template"
    )
