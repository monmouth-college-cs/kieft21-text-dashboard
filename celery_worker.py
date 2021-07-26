from app import celery, create_app, init_celery

#app = create_app(os.getenv('FLASK_CONFIG') or 'default')
#app.app_context().push()

app = create_app()
init_celery(celery, app)

