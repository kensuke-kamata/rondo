import os
import datetime
import subprocess

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)

    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for x in f.inputs:
        txt += dot_edge.format(id(x), id(f))
    for y in f.outputs: # y is weakref
        txt += dot_edge.format(id(f), id(y()))
    return txt

def plot_dot_graph(output, verbose=True, to_file='graph.png'):
    dir = '.graph'
    if not os.path.exists(dir):
        os.makedirs(dir)
    path_dot  = os.path.join(dir, 'graph.dot')
    path_file = os.path.join(dir, to_file)

    dot_graph = get_dot_graph(output, verbose)
    with open(path_dot, 'w') as f:
        f.write(dot_graph)

    ext = os.path.splitext(to_file)[1][1:]
    cmd = 'dot {} -T {} -o {}'.format(path_dot, ext, path_file)
    subprocess.run(cmd, shell=True)

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for x in func.inputs:
            txt += _dot_var(x, verbose)

            if x.creator is not None:
                add_func(x.creator)

    return 'digraph g {\n' + txt + '}'
