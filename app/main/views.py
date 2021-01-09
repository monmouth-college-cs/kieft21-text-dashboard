from flask import render_template, session, redirect, \
    url_for, current_app, flash, Markup, request
from . import main
from .forms import UploadForm, make_wrangle_form, make_analysis_form, SaveForm
from .process import process
from .util import is_supported_file
from werkzeug.utils import secure_filename
from pathlib import Path
import sys, json, zipfile, re
import pandas as pd

def get_datasets():
    path = (Path(current_app.instance_path) / 'datasets' / 'raw').resolve()
    def valid(p):
        if not p.is_dir(): return False
        for bad in '._':
            if str(p.name).startswith(bad): return False
        return True
    dirs = [dset for dset in path.iterdir() if valid(dset)]
    raw = [str(dset.name) for dset in dirs]
    wrangled = [str(dset.name) for dset in dirs if (dset/'.info.json').is_file()]
    return raw, wrangled

def get_dataset_home(dataset, kind='raw'):
    return (Path(current_app.instance_path) / 'datasets' / kind / dataset).resolve()

def get_output_home(dataset, tag):
    return (Path(current_app.instance_path) / 'outputs' / dataset / tag).resolve()

def get_dataset_info(dataset):
    path = get_dataset_home(dataset) / '.info.json'
    with open(path, 'r') as f:
        info = json.load(f)
    return info

# Does not apply to error handlers!
@main.context_processor
def inject_datasets():
    r,w = get_datasets()
    return {'datasets_raw': r, 'datasets_wrangled': w}

def unzip_dataset(name, path, dirname):
    with zipfile.ZipFile(path, 'r') as f:
        f.extractall(dirname)

@main.app_errorhandler(404)
def page_not_found(e):
    r,w = get_datasets()
    return render_template('404.html', datasets_raw=r, datasets_wrangled=w), 404

@main.route('/')
@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@main.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.zipfile.data
        name = form.name.data
        filename = secure_filename(f.filename)
        path = (Path(current_app.instance_path) / 'datasets' / 'zips' / filename).resolve()
        dirname = get_dataset_home(name)
        if dirname.exists():
            flash('A dataset with that name already exists.')
            return redirect(url_for('.upload'))
        
        f.save(path)
        dirname.mkdir(exist_ok=False)
        unzip_dataset(name, path, dirname)
        
        return redirect(url_for('.wrangle', dataset=name))
    return render_template('upload.html', form=form)

def get_analysis_form(dataset):
    info = get_dataset_info(dataset)
    return make_analysis_form(info['level_names'], info['level_vals'])

def do_explore(dataset, form):
    path = get_dataset_home(dataset)
    data = get_dataset_info(dataset)
    for field in form: # assume no subfields?
        if field.name == 'submit' or field.name.startswith('csrf'): continue
        assert field.name not in data
        data[field.name] = field.data

    results = process(dataset, path, data)
    return results

@main.route('/explore/<dataset>/', methods=['GET', 'POST'])
@main.route('/explore/<dataset>/<tag>', methods=['GET', 'POST'])
def explore(dataset, tag='__default'):
    save_form = SaveForm()
    analysis_form = get_analysis_form(dataset)
    results = dict()
    tab = None
    if 'submit_save' in request.form:
        if save_form.validate_on_submit():
            new_tag = save_form.tag.data
            # @TODO: Check for data overwrite when saving results (use
            # checkbox to validate if user wants to)

            # @TODO: copy data from old tag dir to new tag dir, flash
            # message saying you can come back to this url anytime.
            
            flash(f'Not implemented yet; simulate save {tag} -> {new_tag}')
            return redirect(url_for('.explore', dataset=dataset, tag=new_tag))
        else:
            for field, msgs in save_form.errors.items():
                flash(f'Unique name {msgs[0]}')
    elif 'submit' in request.form and analysis_form.validate_on_submit():
        results = do_explore(dataset, analysis_form)
        tab = 'summary'
    return render_template('explore.html', dataset=dataset, tag=tag, active_tab=tab,
                           analysis_form=analysis_form, save_form=save_form,
                           results=results)

def get_levels(dname):
    p = get_dataset_home(dname)
    next_level = [child for child in p.iterdir() if child.is_dir()]
    levels = [{str(child.name) for child in next_level}]
    def clean(p):
        return str(p.name).lower().replace(' ', '_')
    
    while len(next_level) > 0:
        this_level = next_level
        next_level = []
        levels.append(set())
        for p in this_level:
            dirs = [child for child in p.iterdir() if child.is_dir()]
            next_level.extend(dirs)
            levels[-1].update({clean(child) for child in dirs})

    # Seems like this should always be true, but I'm not going to think too hard about it.
    if len(levels[-1]) == 0:
        levels.pop()
    return levels

def human_readable_size(size):
    from math import log2
    order = int(log2(size) / 10) if size else 0
    suffix = ['bytes', *[f"{p}B" for p in 'KMGTPEZY']][order]
    val = size / (1 << (order * 10))
    return f'{val:.2f} {suffix}'

def get_files(dname):
    path = get_dataset_home(dname)

    total_size, valid_size = 0, 0
    files = []
    for child in path.rglob('*'):
        size = child.stat().st_size
        total_size += size
        if is_supported_file(child):
            files.append(str(child.absolute().relative_to(path.absolute())))
            valid_size += size
    return files, human_readable_size(valid_size), human_readable_size(total_size)

def do_wrangle(name, oneper, splitter, use_regex, level_names, level_vals):
    vals = [list(x) for x in level_vals]
    info = {'level_names': level_names, 'level_vals': vals}
    if oneper:
        info['article_regex_splitter'] = None
    elif use_regex:
        info['article_regex_splitter'] = splitter
    else: # plain text splitter, treat as regex anyway
        info['article_regex_splitter'] = re.escape(splitter)
    path = get_dataset_home(name)
    with open(path / '.info.json', 'w') as f:
        json.dump(info, f)
    path = get_output_home(name, '__default')
    if not path.exists():
        path.mkdir(parents=True, exist_ok=False)

    
@main.route('/wrangle/<dataset>', methods=['GET', 'POST'])
def wrangle(dataset):
    files, vsize, tsize = get_files(dataset)
    levels = get_levels(dataset)
    info = {'dataset': dataset, 'files': files, 'levels': levels,
            'vsize': vsize, 'tsize': tsize}
    form = make_wrangle_form(levels)

    if form.validate_on_submit():
        level_names = [getattr(form, field).data for field in dir(form)
                       if field.startswith('level')]
        do_wrangle(dataset, form.oneper.data,
                   form.splitter.data, form.split_regex.data, level_names, levels)
        return redirect(url_for('.explore', dataset=dataset))
    return render_template('wrangle.html', form=form, **info)

