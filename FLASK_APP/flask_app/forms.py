"""Form object declaration."""
from flask_wtf import FlaskForm
from wtforms import (
    DateField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    SelectMultipleField,
    TextAreaField,
    MultipleFileField,
)
from wtforms.validators import URL, DataRequired, Email, EqualTo, Length
# from FLASK_APP.aa import till_display


class VisualizeForm(FlaskForm):
    """Visualize Form"""
    # address_A = StringField("address_A", [DataRequired()])
    # address_B = StringField("address_B", [DataRequired()])
    files_A = MultipleFileField('File(s) Upload A', [DataRequired()])
    files_B = MultipleFileField('File(s) Upload E', [DataRequired()])
    sampling_rate = StringField("sampling_rate", [DataRequired()],default="173.6")
    apply_filter = SelectField(
        "apply_filter",
        [DataRequired()],
        choices=[
            ("yes", "yes"),
            ("no", "no"),
        ],
    )
    low_frequency = StringField("low_frequency", [DataRequired()],default="10")
    # low_frequency = SelectField(
    #     "low_frequency",
    #     [DataRequired()],
    #     choices=[
    #         ("80", "80"),
    #         ("None", "None"),
    #     ],
    # )
    high_frequency = StringField("high_frequency", [DataRequired()],default="80")
    # high_frequency = SelectField(
    #     "high_frequency",
    #     [DataRequired()],
    #     choices=[
    #         ("10", "10"),
    #         ("None", "None"),
    #     ],
    # )
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
    delta_frequency_initial = StringField("delta_frequency_initial", [DataRequired()],default="0.5")
    delta_frequency_final = StringField("delta_frequency_final", [DataRequired()],default="3.5")
    theta_frequency_initial = StringField("theta_frequency_initial", [DataRequired()],default="3.5")
    theta_frequency_final = StringField("theta_frequency_final", [DataRequired()],default="7.5")
    alpha_frequency_initial = StringField("alpha_frequency_initial", [DataRequired()],default="7.5")
    alpha_frequency_final = StringField("alpha_frequency_final", [DataRequired()],default="13")
    beta_frequency_initial = StringField("beta_frequency_initial", [DataRequired()],default="13")
    beta_frequency_final = StringField("beta_frequency_final", [DataRequired()],default="30")
    gamma_frequency_initial = StringField("gamma_frequency_initial", [DataRequired()],default="30")
    gamma_frequency_final = StringField("gamma_frequency_final", [DataRequired()],default="50")
    initial_datapoint = StringField("initial_datapoint", [DataRequired()],default="0")
    final_datapoint = StringField("final_datapoint", [DataRequired()],default="4097")

    submit = SubmitField("Next")


class FeatureForm(FlaskForm):
    """Feature Form"""
    window_size = StringField("window_size", [DataRequired()],default="4097")
    feat_input = SelectMultipleField(
        "feat_input",
        choices=[
            ("median_absolute_deviation", "median_absolute_deviation"),
            ("kurtosis", "kurtosis"),
            ("standard_dev", "standard_dev"),
            ("integration", "integration"),
            ("variance", "variance"),
            ("mean", "mean"),
            # ("app_entropy", "app_entropy"),
            ("hurst_expo", "hurst_expo"),
            ("detrended_fluctuation", "detrended_fluctuation"),
            ("sample_entropy", "sample_entropy"),
            ("correlation_dim", "correlation_dim"),
            ("lyapunov_expo", "lyapunov_expo"),
        ],
    )
    embedded_dim_sample_entropy = StringField("embedded_dim_sample_entropy")
    embedded_dim_correlation_dim = StringField("embedded_dim_correlation_dim")
    embedded_dim_lyapunov_expo = StringField("embedded_dim_lyapunov_expo")
    feat_selection_method = SelectField(
        "feat_selection_method",
        [DataRequired()],
        choices=[
            ("SelectPercentile", "SelectPercentile"),
            ("SelectKBest", "SelectKBest"),
            ("none", "none"),
        ],
    )
    percentage_or_value = StringField("percentage_or_value", [DataRequired()],default="20")
    cv_fold = StringField("cv_fold", [DataRequired()],default="5")
    k_fold_index = SelectField(
        "k_fold_index",
        [DataRequired()],
        choices=[
            ("0", "KFold"),
            ("1", "StratifiedKFold"),
        ],
    )
    classifier_index = SelectField(
        "classifier_index",
        [DataRequired()],
        choices=[
            ("0", "SVC"),
            ("1", "SGDClassifier"),
            ("2", "KNeighborsClassifier"),
            ("3", "GaussianProcessClassifier"),
            ("4", "GaussianNB"),
            ("5", "DecisionTreeClassifier"),
            ("6", "AdaBoostClassifier"),
            ("7", "GradientBoostingClassifier"),
        ],
    )

    run = SubmitField("Run")