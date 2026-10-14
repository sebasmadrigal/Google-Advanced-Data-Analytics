"""Execute plain-Python notebook cells in fresh namespaces.

Capture stdout, DataFrame displays and Matplotlib figures as nbformat v4 outputs.
Run from the repository root. Supported cell content: plain Python.
"""
import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def execute(path):
    notebook = json.loads(path.read_text())
    namespace = {'__name__': '__main__'}
    execution_count = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        execution_count += 1
        print(f'{path.name}: cell {execution_count}', flush=True)
        cell['outputs'] = []
        cell['execution_count'] = execution_count
        outputs = cell['outputs']

        def display(*values):
            for value in values:
                if value is None:
                    continue
                data = {'text/plain': repr(value)}
                if hasattr(value, 'to_html'):
                    data['text/html'] = value.to_html()
                    data['text/plain'] = value.to_string()
                outputs.append({'output_type': 'display_data', 'data': data, 'metadata': {}})

        def show(*args, **kwargs):
            for number in plt.get_fignums():
                fig = plt.figure(number)
                image = io.BytesIO()
                fig.savefig(image, format='png', bbox_inches='tight', dpi=110)
                outputs.append({'output_type': 'display_data',
                                'data': {'image/png': base64.b64encode(image.getvalue()).decode(),
                                         'text/plain': '<Matplotlib figure>'}, 'metadata': {}})
            plt.close('all')

        namespace['display'] = display
        plt.show = show
        buffer = io.StringIO()
        tree = ast.parse(''.join(cell['source']))
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            if tree.body and isinstance(tree.body[-1], ast.Expr):
                last = tree.body.pop()
                exec(compile(tree, str(path), 'exec'), namespace)
                value = eval(compile(ast.Expression(last.value), str(path), 'eval'), namespace)
                if value is not None:
                    display(value)
            else:
                exec(compile(tree, str(path), 'exec'), namespace)
        if buffer.getvalue():
            outputs.insert(0, {'output_type': 'stream', 'name': 'stdout', 'text': buffer.getvalue()})
            print(buffer.getvalue(), end='', flush=True)
        if plt.get_fignums():
            show()
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + '\n')
    return execution_count


if __name__ == '__main__':
    os.chdir(ROOT)
    start = time.perf_counter()
    paths = [ROOT / p for p in sys.argv[1:]] if len(sys.argv) > 1 else sorted((ROOT / 'notebooks').glob('*.ipynb'))
    counts = {p.name: execute(p) for p in paths}
    print(json.dumps({'executed_code_cells': counts, 'seconds': round(time.perf_counter()-start, 1)}, indent=2))
