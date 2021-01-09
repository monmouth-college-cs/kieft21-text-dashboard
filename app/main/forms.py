from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, SubmitField, SelectField, SelectMultipleField, \
    BooleanField, RadioField
from wtforms.validators import DataRequired, Regexp
from flask import Markup
import re


class SaveForm(FlaskForm):
    _rv = Regexp(re.compile(r'^(?!__)\w+$'),
                 message="Must be alphanumeric and not start with double underscores.")
    tag = StringField('Unique Name', validators=[DataRequired(), _rv],
                      description = "Please specify a unique string.")
    submit_save = SubmitField('Save Results')

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


# Help Popovers.
from wtforms import Label
Label_oldcall = Label.__call__
def Label_call(self, text=None, **kwargs):
    extra = ''
    if hasattr(self, 'extra') and self.extra is not None:
        extra = ' ' + self.extra
    return Label_oldcall(self, text, **kwargs) + ' ' + extra
Label.__call__ = Label_call

class MyBool(BooleanField):
    def __init__(self, label='', extra=None, validators=None, **kwargs):
        super(BooleanField, self).__init__(label, validators, **kwargs)
        self.label.extra = extra

# I think you can probably use FieldList from wtforms instead of this,
# but I'm under a time crunch right now and this works.
# 
# @TODO: Use wtforms functionality instead of class factory for wrangle form.
def make_wrangle_form(levels):
    class WrangleForm(FlaskForm):
        helptext = "And here's some amazing content. It's very engaging. Right?"
        popover = f'data-toggle="popover" title="Popover title" data-content="{helptext}"'
        icon = Markup(f'<span class="fas fa-question-circle" {popover}></span>')
        label = Markup(f'One article per file?')
        oneper = MyBool(label, default=True, extra=icon,
                        description='One line description.',
                        render_kw={'onchange': 'toggleSplitter()'})
        splitter = StringField('Article Splitter',
                               description = "Only for plain text files.",
                               render_kw={'disabled': 'true'})
        split_regex = BooleanField('Use Regular Expression?', render_kw={'disabled': 'true'})
        #submit = SubmitField('Wrangle')

    for i, level in enumerate(levels):
        vals = '/'.join(level)
        field = StringField(f'Level {i} name ({vals})', validators=[DataRequired()])
        setattr(WrangleForm, f'level{i}_name', field)

    # This is only here so that wtf.quick_form(...) will lay it out correctly
    # Otherwise the submit button will be above the level names
    setattr(WrangleForm, 'submit', SubmitField('Wrangle'))
    
    return WrangleForm()

# @TODO: Use wtforms functionality instead of class factory for analysis form.
def make_analysis_form(level_names, level_vals):
    unit_choices = [('fixed_windows', 'Fixed-Sized Windows (5 sentences)'),
                    ('paragraphs', 'Paragraphs'),
                    ('articles', 'Articles'),
                    ('context_no_overlap', 'Non-overlapping Contexts (5 sentences)'),
                    ('context_with_overlap', 'Overlapping Contexts (5 sentences)'),
                    ('sentences', 'Sentences'),
    ]
    class AnalysisForm(FlaskForm):
        unit = SelectField('Unit of Analysis', choices=unit_choices)
        
        fterms = StringField('Filter Terms')
        fcase = BooleanField('Ignore case?', default=True)
        fregex = BooleanField('Use Regular Expression?', default=False)
        
        aterms = StringField('Analysis Terms')
        acase = BooleanField('Ignore case?', default=True)
        aregex = BooleanField('Use Regular Expression?', default=False)

    #@TODO: Either have a 'select all' button/checkbox, or select all by default
    for i, (name, vals) in enumerate(zip(level_names, level_vals)):
        c = [(v,v) for v in vals]
        field = SelectMultipleField(name, description="Use ctrl to select multiple",
                                    choices=c)
        setattr(AnalysisForm, f'level{i}_filter', field)

    # @TODO: word2vec vectorizer (slow)
    # Not enough time to implement other options, so leaving this out for now.
    # setattr(AnalysisForm, 'vectorizer',
    #         SelectField('Vectorization', choices=['TFIDF', 'Word Counts']))
    # setattr(AnalysisForm, 'clusterizer',
    #         SelectField('Clustering', choices=['KMeans', 'DBSCAN']))
    # setattr(AnalysisForm, 'submit', SubmitField('Analyze'))

    #@TODO: "Remember these settings" checkbox
    return AnalysisForm()
            
    
