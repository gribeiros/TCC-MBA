import json

notebook_path = 'c:/Users/Gustavo/Documents/TCC-MBA/notebooks/series_exploration.ipynb'
out_path = 'c:/Users/Gustavo/Documents/TCC-MBA/notebook_analysis.json'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

analysis = {}
for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        source = ''.join(c.get('source', []))
        if 'train_log' in source:
            analysis[f'cell_{i}_train_log'] = source
        if 'Bidirectional(LSTM' in source:
            analysis[f'cell_{i}_lstm'] = source

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2)
