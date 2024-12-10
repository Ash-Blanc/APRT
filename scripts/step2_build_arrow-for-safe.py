import json
import os
import glob
import pandas as pd
from datasets import Dataset
import sys
inpath = sys.argv[1]
outpath = sys.argv[2]
if not os.path.exists(outpath):
    os.makedirs(outpath)

files = glob.glob(inpath)

def build_data(line, name):
    new_data = {
        'id': name,
        'conversations': [
            {
                'from': 'human',
                'value': line['instruction']
            },
            {
                'from': 'gpt',
                'value': line['output']
            },
        ]
    }
    return new_data

for file in files:
    filename = file.split('/')[-1]
    dataset_name = '.'.join(filename.split('.')[:-1])
    with open(file, 'r') as f:
        new_content = []
        for i, line in enumerate(f):
            print(line)
            line = json.loads(line)
            new_line = build_data(line, f"{dataset_name}_{i}")
            #new_line = line
            new_content.append(new_line)
        df = pd.DataFrame(new_content)
        dataset = Dataset.from_pandas(df)
        out_arrow_path = f"{outpath}/{dataset_name}/"
        if not os.path.exists(out_arrow_path):
            os.mkdir(out_arrow_path)
        dataset.save_to_disk(out_arrow_path)
