from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, SubmitField, SelectField, SelectMultipleField, \
    BooleanField, RadioField
from wtforms.validators import DataRequired
import re

# @TODO: I would really like to allow uploading a whole folder
# (preserving hierarchy), but it seems difficult. See here:
# https://stackoverflow.com/questions/5826286/how-do-i-use-google-chrome-11s-upload-folder-feature-in-my-own-code
class UploadForm(FlaskForm):
    # @TODO: validate that this name does not already exist. This is
    # already done on the server side, but it can be annoying since it
    # uploads all the data first (I think).
    name  = StringField('Dataset name', validators=[DataRequired()])
    zipfile = FileField('Dataset Zip Files',
                        validators=[FileRequired(), FileAllowed(['zip'], 'Zip files only.')])
    submit = SubmitField('Upload')

    # def validate_zipfile(form, field):
    #     if field.data:
    #         field.data = re.sub(r'[^a-z0-9_.-]', '_', field.data)

# I think you can probably use FieldList from wtforms instead of this,
# but I'm under a time crunch right now and this works.
# 
# @TODO: Use wtforms functionality instead of class factory for wrangle form.
def make_wrangle_form(levels):
    class WrangleForm(FlaskForm):
        oneper = RadioField("One article per file?", choices=['Yes', 'No'])
        splitter = StringField('Article Splitter', render_kw={'disabled': 'true'},
                               validators=[DataRequired()])
        regex = BooleanField('Use Regular Expression?')
        #submit = SubmitField('Wrangle')

    for i, level in enumerate(levels):
        vals = '/'.join(level)
        field = StringField(f'Level {i} name ({vals})', validators=[DataRequired()])
        setattr(WrangleForm, f'level{i}_name', field)

    # This is only here so that wtf.quick_form(...) will lay it out correctly
    # Otherwise the submit button will be above the level names
    setattr(WrangleForm, 'submit', SubmitField('Wrangle'))
    
    return WrangleForm()
        

# @TODO: Remove old stuff
class NameForm(FlaskForm):
    name = StringField('What is your name?', validators=[DataRequired()])
    submit = SubmitField('Submit')

class MyForm(FlaskForm):
    unit = SelectField('Unit of Analysis', choices=['Sentence', 'Paragraph', 'Windows'])
    fterms = StringField('Filter Terms')
    aterms = StringField('Analysis Terms')

    # @TODO: Make 'levels' dynamic, based on folder names
    c = [('US','United States'), ('MX','Mexico'), ('NT','Northern Triangle')]
    regions = SelectMultipleField('Region', choices=c)
    c = [(i, str(i)) for i in range(1,5)]
    periods = SelectMultipleField('Period', choices=c)

    # @TODO: word2vec vectorizer (slow)
    vectorizer = SelectField('Vectorization', choices=['TFIDF', 'Word Counts'])
    clusterizer = SelectField('Clustering', choices=['KMeans', 'DBSCAN'])
    
    name = StringField('What is your name?', validators=[DataRequired()])
    submit = SubmitField('Submit')
    
