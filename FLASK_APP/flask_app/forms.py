"""Form object declaration."""
from flask_wtf import FlaskForm
from wtforms import (
    DateField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import URL, DataRequired, Email, EqualTo, Length
# from FLASK_APP.aa import till_display


class VisualizeForm(FlaskForm):
    """Visualize Form"""
    address_A = StringField("address_A", [DataRequired()])
    address_B = StringField("address_B", [DataRequired()])
    sampling_rate = StringField("sampling_rate", [DataRequired()])
    low_frequency = SelectField(
        "low_frequency",
        [DataRequired()],
        choices=[
            ("80", "80"),
            ("None", "None"),
        ],
    )
    high_frequency = SelectField(
        "high_frequency",
        [DataRequired()],
        choices=[
            ("10", "10"),
            ("None", "None"),
        ],
    )
    filter_method = SelectField(
        "filter_method",
        [DataRequired()],
        choices=[
            ("fir", "fir"),
            ("iir", "iir"),
        ],
    )
    window_type = SelectField(
        "window_type",
        [DataRequired()],
        choices=[
            ("hamming", "hamming"),
            ("hann", "hann"),
            ("blackman", "blackman"),
        ],
    )
    # submit = SubmitField("Next", render_kw={"onclick": ""})
    submit = SubmitField("Next")