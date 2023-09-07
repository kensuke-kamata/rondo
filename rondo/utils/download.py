import os
import urllib.request

def download(url, filename=None):
    if filename is None:
        filename = url.split('/')[-1]

    cachepath = os.path.join(os.getcwd(), '.cache')
    if not os.path.exists(cachepath):
        os.mkdir(cachepath)

    filepath = os.path.join(cachepath, filename)
    if os.path.exists(filepath):
        return filepath

    print(f'Downloading {filename}')
    try:
        urllib.request.urlretrieve(url, filepath, progress)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise e
    print('Done')

    return filename

def progress(block_num, block_size, total_size):
    p = min(1.0, float(block_num * block_size) / total_size)
    bar = '#' * int(p * 100)
    print(f'\r[{bar:<100}] {p:.2%}', end='')
    if p == 1.0:
        print()
