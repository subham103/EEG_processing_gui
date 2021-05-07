"""Initialize app"""
from flask import Flask

def create_app():
    """Construct the core flask_app"""
    app = Flask(__name__, instance_relative_config=False)
    app.config.from_object("config.Config")
    app.config["RECAPTCHA_PUBLIC_KEY"] = "dfghjsujhsgtdhikdnd"
    app.config["RECAPTCHA_PARAMETERS"] = {"size": "100%"}

    with app.app_context():
        from . import routes

        return app
