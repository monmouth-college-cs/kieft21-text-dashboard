import os, logging, sys

from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_fontawesome import FontAwesome
from flask_dropzone import Dropzone

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
    
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'bti.sqlite'),
        MAX_CONTENT_LENGTH=95*1024*1024, # 95 mib
    )

    if test_config is None: # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else: # load the test config if passed in
        app.config.from_mapping(test_config)

    try: # ensure the necessary folders exist
        datasets = os.path.join(app.instance_path, 'datasets')
        os.makedirs(datasets, exist_ok=True)
        os.makedirs(os.path.join(datasets, 'zips'), exist_ok=True)
        os.makedirs(os.path.join(datasets, 'raw'), exist_ok=True)

        os.makedirs(os.path.join(app.instance_path, 'outputs'), exist_ok=True)
    except OSError as e:
        print(e)
        pass

    bootstrap = Bootstrap(app)
    fa = FontAwesome(app)
    dropzone = Dropzone(app)

    from .main import main as main_blueprint
    app.register_blueprint(main_blueprint)
    
    return app
