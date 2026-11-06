"""Look up the fixed native Judge/Guard/preference-model snapshot revision."""
import argparse
import json
from pathlib import Path

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model')
    p.add_argument('source', choices=['HF', 'ModelScope'])
    args = p.parse_args()
    data = json.loads((Path(__file__).resolve().parents[1] / 'configs/cross_model_twelve_dependency_models.json').read_text())
    print(data[args.model][args.source + '_revision'])
