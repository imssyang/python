import logging
import os
from flask import Flask


def create_app():
    app = Flask(__name__)

    from api import flow
    app.register_blueprint(flow.bp)

    pypath = os.environ["PYTHONPATH"]
    logging.info(f"PYTHONPATH: {pypath}")
    #logging.info(f"app: {app.__dict__}")
    return app
