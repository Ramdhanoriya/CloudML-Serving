import sys

is_python2 = int(sys.version[0]) == 2
is_windows = sys.platform.startswith('win')
is_linux = sys.platform.startswith('linux')
is_osx = sys.platform == 'darwin'

zip_ = zip
bytes_ = bytes
unicode_ = str
string_types = (bytes, str)
chr_ = chr


def unicode_to_bytes(s, encoding='utf8', errors='strict'):
    return s.encode(encoding=encoding, errors=errors)


def bytes_to_unicode(b, encoding='utf8', errors='strict'):
    return b.decode(encoding=encoding, errors=errors)


def get_config():
    """Helper function to get relevant config info, especially when debugging."""
    from spacy.about import __version__ as spacy_version
    from spacy.util import get_data_path

    return {
        'python': sys.version,
        'platform': sys.platform,
        'spacy': spacy_version,
        'spacy_models': [
            d.parts[-1] for d in get_data_path().iterdir()
            if (d.is_dir() or d.is_symlink()) and
               d.parts[-1] not in {'__cache__', '__pycache__'}]
    }
