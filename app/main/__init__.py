from flask import Blueprint

import nltk
nltk.download('punkt')

main = Blueprint('main', __name__)

# Must be at end -- circular dependency
from . import views #, errors
