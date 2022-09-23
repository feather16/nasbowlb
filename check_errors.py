import os
import re

LOG_DIR = f'{os.path.dirname(__file__)}/result/log'

found_errors = 0
files = sorted(os.listdir(LOG_DIR))
for file in files:
    if re.fullmatch(r'err_\d+.out', file):
        with open(f'{LOG_DIR}/{file}') as f:
            text = f.read()
            if len(text) > 0:
                print(f'Error found: {file} {text[:20]} ...')
                found_errors += 1
if found_errors == 0:
    print('No errors found!')