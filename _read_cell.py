import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

nb_file = sys.argv[1]
cell_idx = int(sys.argv[2])

with open(nb_file, 'r', encoding='utf-8') as f:
    nb = json.load(f)

src = ''.join(nb['cells'][cell_idx]['source'])
print(src)
