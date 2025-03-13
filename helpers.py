import os

def get_project_root_dir():
    return os.path.dirname(os.path.abspath(__file__))

def print_with_indent(*to_print, n_indent: int, **kwargs):
    print('\t' * n_indent, end='')
    print(*to_print, **kwargs)