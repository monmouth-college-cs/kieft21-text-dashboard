import sys, json, zipfile, re, os, uuid
from pathlib import Path

from flask import render_template, session, redirect, \
    url_for, current_app, flash, Markup, request, jsonify
from flask_socketio import emit, join_room, leave_room
from werkzeug.utils import secure_filename
from werkzeug.datastructures import MultiDict as FormDataStructure
from wtforms import FormField
import pandas as pd

from . import main
from .. import socketio
from .forms import UploadForm, make_wrangle_form, make_analysis_form, U2Form
from .process import process, wrangle_dataset, make_dataset, build_default_stopwords
from .util import is_supported_file, get_datasets, get_dataset_home, get_output_home, \
    get_dataset_info

# Does not apply to error handlers!
@main.context_processor
def inject_datasets():
    r,p,w = get_datasets()
    return {'datasets_raw': r, 'datasets_preprocessed': p, 'datasets_wrangled': w}

@main.app_errorhandler(404)
def page_not_found(e):
    r,p,w = get_datasets()
    return render_template('404.html', datasets_raw=r,
                           datasets_preprocessed=p, datasets_wrangled=w), 404

@main.route('/robots.txt')
def robots():
    return "User-agent: *\nDisallow: /"

@main.route('/')
@main.route('/about')
def about():
    return render_template('about.html')

@main.route('/tutorial')
def tutorial():
    return render_template('tutorial.html')

@main.route('/test', methods=['GET', 'POST'])
def test():
    form = U2Form()
    if form.validate_on_submit():
        msg = f"Got files: {form.files.data}\n\nBut directory structure is lost! Need to use Javascript and write custom WTForms class"
        flash(msg)
    return render_template('test.html', form=form)

@main.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.zipfile.data
        filename = secure_filename(f.filename)
        name = form.name.data.strip() or os.path.splitext(filename)[0]
        path = (Path(current_app.instance_path) / 'datasets' / 'zips' / filename).resolve()
        dirname = get_dataset_home(name)
        if dirname.exists():
            flash('A dataset with that name already exists.')
            return redirect(url_for('.upload'))
        
        f.save(path)
        finished = make_dataset(dirname, path)
        if finished:
            return redirect(url_for('.wrangle', dataset=name))
        flash("Data uploaded and preprocessing started. It may take a while.")
        return redirect(url_for('.inprogress', dataset=name, stage='preprocess'))
    return render_template('upload.html', form=form)

def get_analysis_form(dataset, formdata=None):
    info = get_dataset_info(dataset)
    return make_analysis_form(info['level_names'], info['level_vals'], formdata)

def do_explore(dataset, form, uid):
    path = get_dataset_home(dataset)
    data = get_dataset_info(dataset)
    for field in form: # assume no subfields?
        if field.name == 'submit' or field.name.startswith('csrf'): continue
        assert field.name not in data
        if isinstance(field, FormField):
            for subfield in field:
                data[subfield.name] = subfield.data
        else:
            data[field.name] = field.data

    return process(dataset, path, data, uid)

def load_results(path, tag):
    outfile = path / '.output' / tag
    if not outfile.exists():
        return None
    with open(outfile, 'r') as f:
        results = json.load(f)
    return results

@main.route('/analysisform', methods=['POST'])
def on_analysis_submit():
    data = request.json
    dataset = request.json['dataset']
    analysis_form = get_analysis_form(dataset)
    if 'formdata' in data:
        form_data = FormDataStructure(data['formdata'])
    analysis_form = get_analysis_form(dataset, formdata=form_data)

    if not analysis_form.validate():
        return jsonify(data={'form error': analysis_form.errors}), 400
    
    roomid = str(uuid.uuid4())
    details = [dataset, roomid]

    task = do_explore(dataset, analysis_form, roomid)

    return jsonify({'roomid': roomid}), 202


@main.route('/explore/<dataset>/', methods=['GET', 'POST'])
@main.route('/explore/<dataset>/<tag>', methods=['GET', 'POST'])
def explore(dataset, tag=None):
    path = get_dataset_home(dataset)
    if not path.exists():
        flash(f'Dataset "{dataset}" does not exist, but you can upload a new one.')
        return redirect(url_for('.upload'))

    preprocessed = path / '.preprocessed'
    if not preprocessed.exists():
        flash('This dataset is either still preprocessing or there is an error.')
        return redirect(url_for('.inprogress', dataset=dataset, stage='preprocess'))

    wrangled = path / '.info.json'
    if not wrangled.exists():
        flash('This dataset is either still wrangling or there is an error.')
        return redirect(url_for('.inprogress', dataset=dataset, stage='wrangle'))

    results = None
    tab = 'settings'
    analysis_form = get_analysis_form(dataset)

    # Let's try to load the results
    if tag: 
        try:
            results = load_results(path, tag)
        except json.decoder.JSONDecodeError as e:
            flash("Error loading results; something went wrong. Likely a bug.", 'error')
            return redirect(url_for('.explore', dataset=dataset))

        # @TODO: check whether or not the tag just does not exist, or
        # if the computation has started but not finished.
        if results is None:
            flash("In progress or tag does not exist.");
        elif 'error' in results:
            flash(results['error'], 'error')
        else:
            tab = 'summary'
        
    return render_template('explore.html', dataset=dataset, tag=tag, active_tab=tab,
                           analysis_form=analysis_form, results=results,
                           swords = ", ".join(build_default_stopwords()))

def get_levels(dname):
    p = get_dataset_home(dname)
    next_level = [child for child in p.iterdir()
                  if child.is_dir() and not child.name.startswith('.')]
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
    # Probably should do this more carefully (recursively) to avoid .hidden directories.
    for child in path.rglob('*'):
        size = child.stat().st_size
        total_size += size

        # anything in a directory that starts with '.' should be skipped
        skip = False
        for dirname in str(child.absolute()).split(os.sep):
            if dirname.startswith('.'):
                skip = True
                break
        if skip: continue
        
        if is_supported_file(child):
            files.append(str(child.absolute().relative_to(path.absolute())))
            valid_size += size
    return files, human_readable_size(valid_size), human_readable_size(total_size)

                        
@main.route('/inprogress/<dataset>/<stage>')
@main.route('/inprogress/<dataset>/<stage>/<tag>')                        
def inprogress(dataset, stage, tag=None):
    path = get_dataset_home(dataset)
    if not path.exists():
        flash(f'Dataset "{dataset}" does not exist, but you can upload a new one.')
        return redirect(url_for('.upload'))

    error = path / '.error'
    if error.exists():
        flash(f'Sorry, there was an error during the {stage} stage.', 'error')
        with open(error, 'r') as f:
            flash(f.read(), 'error')

    preprocessed = path / '.preprocessed'
    if stage == 'preprocess' and preprocessed.exists():
        flash('Preprocessing completed!')
        return redirect(url_for('.wrangle', dataset=dataset))

    if stage == 'wrangle':
        assert preprocessed.exists()
        wrangled = path / '.info.json'
        if wrangled.exists():
            return redirect(url_for('.explore', dataset=dataset))

    if stage == 'explore' and tag:
        path = get_dataset_home(dataset)
        outfile = path / '.output' / tag
        if outfile.exists():
            return redirect(url_for('.explore', dataset=dataset, tag=tag))
    
    return render_template('inprogress.html', dataset=dataset, stage=stage)
    
@main.route('/wrangle/<dataset>', methods=['GET', 'POST'])
def wrangle(dataset):
    path = get_dataset_home(dataset)
    if not path.exists():
        flash(f'Dataset "{dataset}" does not exist, but you can upload a new one.')
        return redirect(url_for('.upload'))

    preprocessed = path / '.preprocessed'
    if not preprocessed.exists():
        flash('This dataset is either still preprocessing or there is an error.')
        return redirect(url_for('.inprogress', dataset=dataset, stage='preprocess'))

    preprocessed = path / '.preprocessed'
    if not preprocessed.exists():
        flash('This dataset is either still preprocessing or there is an error.')
        return redirect(url_for('.inprogress', dataset=dataset, stage='preprocess'))
    
    files, vsize, tsize = get_files(dataset)
    levels = get_levels(dataset)
    info = {'dataset': dataset, 'files': files, 'levels': levels,
            'vsize': vsize, 'tsize': tsize}
    form = make_wrangle_form(levels)

    if form.validate_on_submit():
        level_names = [getattr(form, field).data for field in dir(form)
                       if field.startswith('level')]
        finished = wrangle_dataset(path, form.oneper.data, form.splitter.data,
                                   form.split_regex.data, level_names, levels)
        if finished:
            return redirect(url_for('.explore', dataset=dataset))
        flash("Wrangling started. It may take a while.")
        return redirect(url_for('.inprogress', dataset=dataset, stage='wrangle'))

    return render_template('wrangle.html', form=form, **info)

@socketio.on('status')
def bounce_status(message):
    emit('status', {'status': message['status']}, to=request.sid)

@socketio.on('connect')
def connect(data=None):
    print(f'Client connected.')

@socketio.on('join_room')
def join_task_room(data):
    assert 'roomid' in data
    join_room(data['roomid'])
    socketio.emit('status', {'status': f'Joined room: {data["roomid"]}'}, to=data['roomid'])
