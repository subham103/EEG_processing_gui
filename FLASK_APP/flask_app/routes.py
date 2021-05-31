"""Routing."""
from flask import current_app as app
from flask import redirect, render_template, url_for
import sys
import json
from flask.globals import request, session
from werkzeug.utils import secure_filename
import os

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
        # files_filenames = []
        for file in form.files_A.data:
            file_filename = secure_filename(file.filename)
            script_dir = os.path.dirname(__file__)
            # print(os.path.join(script_dir + '/' + app.config['UPLOAD_FOLDER'] + '/A',file_filename), file=sys.stderr)
            file.save(os.path.join(script_dir + '/' + app.config['UPLOAD_FOLDER'] + '/A',file_filename))
            # files_filenames.append(file_filename)
        # print(files_filenames)

        for file in form.files_B.data:
            file_filename = secure_filename(file.filename)
            script_dir = os.path.dirname(__file__)
            file.save(os.path.join(script_dir + '/' + app.config['UPLOAD_FOLDER'] + '/E',file_filename))
        
        # with open('visualize_dump.json', 'w') as f:
        #     json.dump(form.data, f)
        # del form.data['files_A']
        # del form.data['files_B']
        # form.data.pop('files_A')
        # some_val = form.data.pop('files_B')
        visualize_data = {}
        for key in form.data:
            if key!="files_A" and key!="files_B":
                visualize_data[key]=form.data[key]
        till_display(visualize_data)
        # print(some_val, file=sys.stderr)
        print(visualize_data, file=sys.stderr)
        session["visualize_data"] = visualize_data
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

# @app.route('/upload',methods = ['GET','POST'])
# def upload_file():
#     if request.method =='POST':
#         file = request.files.getlist("file[]")
#         print(len(file), file=sys.stderr)
#         for ele in file:
#             if ele:
#                 filename = secure_filename(ele.filename)
#                 script_dir = os.path.dirname(__file__)
#                 ele.save(os.path.join(script_dir + '/' + app.config['UPLOAD_FOLDER'],filename))
#         return redirect(url_for("success"))
#     return render_template('file_upload.html')