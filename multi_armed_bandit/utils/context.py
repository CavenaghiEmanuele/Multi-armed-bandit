import json

from typing import Dict

def from_dict_to_str(state:Dict) -> str:
    return json.dumps(sorted(state.items()))

def from_str_to_dict(state:str) -> Dict:

    return