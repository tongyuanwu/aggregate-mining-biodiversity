import sys, traceback, nbformat
import matplotlib; matplotlib.use('Agg')

def run(nb_path, stop_after):
    nb = nbformat.read(nb_path, as_version=4)
    ns = {'__name__': '__main__'}
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        cid = cell.get('id', '')
        try:
            exec(cell.source, ns)
        except Exception:
            print(f'\n!!! {nb_path} cell {cid} FAILED:', flush=True); traceback.print_exc(); return False
        if cid == stop_after:
            break
    print(f'OK {nb_path} (through {stop_after})', flush=True)
    return True

ok1 = run('Fig-4.ipynb', 'd88a5c81')      # Figure-4.jpg + Extended-Data-Figure-6.jpg
ok2 = run('Fig-3-new.ipynb', '9afe8332')  # Figure-3.jpg (reverted, 5-panel)
sys.exit(0 if (ok1 and ok2) else 1)
