from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms import StringField, SubmitField, SelectField, SelectMultipleField, \
    BooleanField, RadioField, Form, FormField, MultipleFileField
from wtforms.fields import html5 as h5fields
from wtforms.widgets import html5 as h5widgets
from wtforms.validators import DataRequired, Regexp, NumberRange
from flask import Markup
from sklearn.feature_extraction import text
import re

class MyIntegerField(h5fields.IntegerField):
    def __init__(self, label=None, min=None, max=None, step=None, validators=None, **kwargs):
        validators = validators if validators else []
        validators.append(NumberRange(min=min, max=max,
                                      message=f'Must be between {min} and {max}'))
        super().__init__(label, validators,
                         widget=h5widgets.NumberInput(min=min, max=max, step=step),
                         **kwargs)

# Help Popovers.
from wtforms import Label
Label_oldcall = Label.__call__
def Label_call(self, text=None, **kwargs):
    extra = ''
    if hasattr(self, 'extra') and self.extra is not None:
        extra = ' ' + self.extra
    return Label_oldcall(self, text, **kwargs) + ' ' + extra
Label.__call__ = Label_call

def make_popover(helptext, title='Help'):
    popover = f'data-toggle="popover" title="{title}" data-content="{helptext}" placement="auto"'
    icon = Markup(f'<span class="fas fa-question-circle text-primary" {popover}></span>')
    return icon

# class MyBool(BooleanField):
#     def __init__(self, label='', helptext=None, validators=None, **kwargs):
#         super().__init__(label, validators, **kwargs)
#         self.label.extra = helptext

for parent in [BooleanField, StringField, FileField, SelectField]:
    class Class_(parent):
        def __init__(self, label='', validators=None,
                     helptext=None, helptitle='Help', **kwargs):
            super().__init__(label, validators, **kwargs)
            self.label.extra = make_popover(helptext, helptitle)
    globals()['H'+parent.__name__] = Class_

class FolderField(MultipleFileField):
    def __init__(self, label='', validators=None, **kwargs):
        dir_attrs = {'directory': "", 'webkitdirectory': "", 'mozdirectory': ""}
        if 'render_kw' not in kwargs:
            kwargs['render_kw'] = dir_attrs
        else:
            kwargs['render_kw'].update(dir_attrs)
        super().__init__(label, validators, **kwargs)

class U2Form(FlaskForm):
    files = FolderField('Folder?')
    submit = SubmitField('Upload')

# @TODO: I would really like to allow uploading a whole folder
# (preserving hierarchy), but it seems difficult. See here:
# https://stackoverflow.com/questions/5826286/how-do-i-use-google-chrome-11s-upload-folder-feature-in-my-own-code
class UploadForm(FlaskForm):
    # @TODO: validate that this name does not already exist. This is
    # already done on the server side, but it can be annoying since it
    # uploads all the data first (I think).
    name  = HStringField('Dataset name',
                         helptext="This will be used to refer to the dataset permanently. If you don't enter a name, the Zip filename will be used.")
    zipfile = HFileField('Dataset Zip File', helptext="A zip file, as specified in the tutorial.",
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
    help_oneper = "Does each plain text file contain only a single article? Does not apply to Excel files."
    help_splitter = "This pattern will be used to split plain text files into a series of articles. Does not apply to Excel files."
    class WrangleForm(FlaskForm):
        label = Markup(f'One article per file?')
        oneper = HBooleanField(label, default=True, helptext=help_oneper,
                        render_kw={'onchange': 'toggleSplitter()'})
        splitter = HStringField('Article Splitter', helptext=help_splitter,
                                description = "Only for plain text files.",
                                render_kw={'disabled': 'true'})
        split_regex = HBooleanField('Use Regular Expression?',
                                    helptext="Allows more complicated patterns, as described in the tutorial.",
                                    render_kw={'disabled': 'true'})
        #submit = SubmitField('Wrangle')

    help_levels = "Provide a descriptive name for this level."
    for i, level in enumerate(levels):
        vals = '/'.join(level)
        field = HStringField(f'Level {i} name ({vals})', helptext=help_levels,
                             validators=[DataRequired()])
        setattr(WrangleForm, f'level{i}_name', field)

    # This is only here so that wtf.quick_form(...) will lay it out correctly
    # Otherwise the submit button will be above the level names
    setattr(WrangleForm, 'submit', SubmitField('Wrangle'))
    
    return WrangleForm()

def make_level_select(level_names, level_vals):
    class LevelSelect(Form):
        pass
    for i, (name, vals) in enumerate(zip(level_names, level_vals)):
        c = [(v,v) for v in vals]
        field = SelectMultipleField(name, description="Use ctrl to select multiple",
                                    choices=c, default=vals,
                                    validators=[DataRequired()])
        setattr(LevelSelect, f'level{i}_filter', field)
    return FormField(LevelSelect)
    

# @TODO: Use wtforms functionality instead of class factory for analysis form.
def make_analysis_form(level_names, level_vals, formdata=None):
    unit_choices = [('fixed_windows', 'Fixed-Sized Windows (5 sentences)'),
                    ('paragraphs', 'Paragraphs'),
                    ('articles', 'Articles'),
                    ('context_no_overlap', 'Non-overlapping Contexts (5 sentences)'),
                    ('context_with_overlap', 'Overlapping Contexts (5 sentences)'),
                    ('sentences', 'Sentences'),
    ]
    class AnalysisForm(FlaskForm):
        unit = HSelectField('Unit of Analysis', choices=unit_choices,
                            helptext="Determines how we break articles into smaller pieces, called 'chunks'.")

        swords = HStringField('Stop Words', helptext="These are words you can choose to keep out of analysis, to prevent extra noise from affecting reuslts.")
        defaultswords = HBooleanField('Use/add to the default stop words?', default=True, helptext="If you would prefer to use the default stop words list, or add to it, select this box. Click the button below to look at the default stop words.")

        fterms = HStringField('Filter Terms', helptext="Only chunks containing at least one of these terms will be kept. Separate terms with a space.")
        fcase = HBooleanField('Ignore case?', default=True,
                              helptext="If checked, filtering is not case sensitive, i.e., captialized vs not capitalized doesn't amtter.")
        fregex = HBooleanField('Use Regular Expression?', default=False,
                               helptext="Allows more complicated patterns, as described in the tutorial.",)
        
        aterms = HStringField('Analysis Terms', helptext="We analyze the sentiment for all chunks containing these terms. We do this analysis separately for each term, allowing you to compare overall sentiment of each term.")
        acase = HBooleanField('Ignore case?', default=True,
                              helptext="If checked, filtering is not case sensitive, i.e., captialized vs not capitalized doesn't amtter.")
        aregex = HBooleanField('Use Regular Expression?', default=False,
                               helptext="Allows more complicated patterns, as described in the tutorial.",)

    AnalysisForm.level_select = make_level_select(level_names, level_vals)

    # @TODO: word2vec vectorizer (slow)
    # Not enough time to implement other options, so leaving this out for now.
    # setattr(AnalysisForm, 'vectorizer',
    #         SelectField('Vectorization', choices=['TFIDF', 'Word Counts']))
    # setattr(AnalysisForm, 'clusterizer',
    #         SelectField('Clustering', choices=['KMeans', 'DBSCAN']))
    n_clusters = MyIntegerField(label='Number of Clusters', min=1, max=20, default=6)
    AnalysisForm.n_clusters = n_clusters
                 
    AnalysisForm.submit = SubmitField('Analyze',
                                      render_kw={'class': 'btn-lg btn-block'})


    #@TODO: "Remember these settings" checkbox
    
    return AnalysisForm(formdata)
            
    
