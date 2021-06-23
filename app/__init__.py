import os, logging, sys

from flask import Flask, render_template
from flask_bootstrap import Bootstrap
from flask_fontawesome import FontAwesome
from flask_dropzone import Dropzone
from flask_socketio import SocketIO
from celery import Celery
from pathlib import Path

def init_celery(celery, app):
    celery.conf.update(app.config)
    TaskBase = celery.Task
    class ContextTask(TaskBase):
        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)
    celery.Task = ContextTask

def make_celery(app_name=__name__):
    backend = 'redis://localhost:6379/0'
    broker = backend.replace('0', '1')
    return Celery(app_name, backend=backend, broker=broker,
                  include=['app.main.process'])

celery = make_celery()
socketio = SocketIO()

def create_app(app_name=__name__, test_config=None, **kwargs):
    app = Flask(app_name, instance_relative_config=True)
    if kwargs.get("celery"):
        init_celery(kwargs.get("celery"), app)
    
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

    (Path(app.instance_path) / 'outputs').mkdir(parents=True, exist_ok=True)

    socketio.init_app(app, message_queue='redis://localhost:6379/0')

    
    return app
