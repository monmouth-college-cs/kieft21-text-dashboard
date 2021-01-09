from magic import Magic # detect plaintext files
from pathlib import Path

def is_supported_file(filename):
    p = Path(filename) # might already by Path, but ok.
    m = Magic(mime=True)
    if not p.is_file(): return False
    for bad in '._':
        if p.name.startswith(bad): return False
    if m.from_file(str(p.absolute())) == 'text/plain': return True
    if p.name.endswith('.xlsx'): return True
    return False

