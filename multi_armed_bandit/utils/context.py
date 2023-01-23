import json

from typing import Dict

def from_dict_to_str(context:Dict) -> str:
    return json.dumps(sorted(context.items()))

def from_str_to_dict(context:str) -> Dict:

    return