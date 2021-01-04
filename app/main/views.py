from flask import render_template, session, redirect, url_for, current_app, flash
from . import main
from .forms import NameForm, MyForm, UploadForm, make_wrangle_form
from werkzeug.utils import secure_filename
import zipfile
from pathlib import Path
import sys
from magic import Magic # detect plaintext files

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
    return render_template('404.html', **get_datasets()), 404

@main.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadForm()
    if form.validate_on_submit():
        f = form.zipfile.data
        name = form.name.data
        filename = secure_filename(f.filename)
        path = (Path(current_app.instance_path) / 'datasets' / 'zips' / filename).resolve()
        dirname = (Path(current_app.instance_path) / 'datasets' / 'raw' / name).resolve()
        if dirname.exists():
            flash('A dataset with that name already exists.')
            return redirect(url_for('.upload'))
        
        f.save(path)
        dirname.mkdir(exist_ok=False)
        unzip_dataset(name, path, dirname)
        
        return redirect(url_for('.wrangle', dataset=name))
    return render_template('upload.html', form=form)


@main.route('/explore', methods=['GET', 'POST'])
def explore(dataset):
    form = MyForm() # AnalysisForm
    if form.validate_on_submit():
        session['name'] = form.name.data
        return redirect(url_for('.explore'))
    return render_template('explore.html', dataset=dataset,
                           form=form, name=session.get('name'))

def get_levels(dname):
    path = (Path(current_app.instance_path) / 'datasets' / 'raw').resolve()
    p = path / dname
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
    path = (Path(current_app.instance_path) / 'datasets' / 'raw' / dname).resolve()
    f = Magic(mime=True)
    def valid(p):
        if not p.is_file(): return False
        for bad in '._':
            if p.name.startswith(bad): return False
        if f.from_file(str(p.absolute())) == 'text/plain': return True
        if p.name.endswith('.xlsx'): return True
        return False

    total_size, valid_size = 0, 0
    files = []
    for child in path.rglob('*'):
        size = child.stat().st_size
        total_size += size
        if valid(child):
            files.append(str(child.absolute().relative_to(path.absolute())))
            valid_size += size
    return files, human_readable_size(valid_size), human_readable_size(total_size)

def do_wrangle(splitter, use_regex, levels):
    print("Wrangle!")
    print(splitter)
    print(use_regex)
    print(levels)
    
@main.route('/wrangle/<dataset>', methods=['GET', 'POST'])
def wrangle(dataset):
    files, vsize, tsize = get_files(dataset)
    levels = get_levels(dataset)
    info = {'dataset': dataset, 'files': files, 'levels': levels,
            'vsize': vsize, 'tsize': tsize}
    form = make_wrangle_form(levels)

    if form.validate_on_submit():
        level_names = [getattr(form, field).data for field in dir(form) if field.startswith('level')]
        do_wrangle(form.splitter.data, form.regex.data, level_names)
        #return redirect(url_for('.explore', dataset=dataset))        
    return render_template('wrangle.html', form=form, **info)

