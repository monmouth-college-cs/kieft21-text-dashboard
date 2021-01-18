from magic import Magic # detect plaintext files
from pathlib import Path
import json
from flask import current_app

def get_datasets():
    path = (Path(current_app.instance_path) / 'datasets' / 'raw').resolve()
    def valid(p):
        if not p.is_dir(): return False
        for bad in '._':
            if str(p.name).startswith(bad): return False
        return True
    dirs = [dset for dset in path.iterdir() if valid(dset)]
    raw = [str(dset.name) for dset in dirs]
    preprocessed = [str(dset.name) for dset in dirs if (dset/'.preprocessed').is_file()]
    wrangled = [str(dset.name) for dset in dirs if (dset/'.info.json').is_file()]
    return raw, preprocessed, wrangled

def get_dataset_home(dataset, kind='raw'):
    return (Path(current_app.instance_path) / 'datasets' / kind / dataset).resolve()

def get_output_home(dataset, tag):
    return (Path(current_app.instance_path) / 'outputs' / dataset / tag).resolve()

def get_dataset_info(dataset):
    path = get_dataset_home(dataset) / '.info.json'
    with open(path, 'r') as f:
        info = json.load(f)
    return info

# This only checks the filename. Used for extracting from a zip file,
# where the file does not exist on disk yet!
def is_supported_filename(filename):
    p = Path(filename) # might already be Path, but ok.
    name = p.name

    # I forget why I'm disallowing _ now...maybe I wanted to use it for something?
    for bad_prefix in '._':
        if name.startswith(bad_prefix): return False

    # It's a little strict to force .txt extension for plain text, but
    # also dangerous to assume another extension (or not extension) is
    # plain text.
    for good_suffix in ['.xlsx', '.txt']:
        if name.endswith(good_suffix): return True

    # Pessimistic: we probably can't handle it.
    return False

# The file must actually exist on disk!
def is_supported_file(filename):
    p = Path(filename) # might already be Path, but ok.
    m = Magic(mime=True)
    if not p.is_file(): return False
    for bad in '._':
        if p.name.startswith(bad): return False

    # We're a little lax here; anything with plain text is ok, even
    # though it might actually have some structure that we won't
    # detect.
    if m.from_file(str(p.absolute())) == 'text/plain': return True
    if p.name.endswith('.xlsx'): return True
    return False

