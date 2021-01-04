from flask import Blueprint

main = Blueprint('main', __name__)

# Must be at end -- circular dependency
from . import views #, errors
