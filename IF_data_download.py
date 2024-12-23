"""
IF_data_download.py

Downloading and processing instructio tuning data of CoinMath.
"""

import json
import os
from datasets import load_dataset, concatenate_datasets

def save_jsonl(data, output_path):
    # create folder if not exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as fout:
        for example in data:
            fout.write(json.dumps(example) + "\n")

ds = load_dataset("amao0o0/CoinMath")

data_ensemble1 = concatenate_datasets([ds['concise_comment'], ds['descriptive'], ds['hardcoded']])
data_ensemble2 = concatenate_datasets([ds['no_comment'], ds['obscure'], ds['general']])
data_ensemble3 = concatenate_datasets([ds['no_comment'], ds['concise_comment'], ds['detailed_comment'], ds['descriptive'], ds['obscure'], ds['general'], ds['hardcoded']])

save_jsonl(data_ensemble1, "IF_data/conciseComment-descriptive-hardcoded.jsonl")
save_jsonl(data_ensemble2, "IF_data/noComment-obscure-general.jsonl")
save_jsonl(data_ensemble3, "IF_data/allTypes.jsonl")