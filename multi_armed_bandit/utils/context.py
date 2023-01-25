import json
import pandas as pd

from typing import Dict

def from_dict_to_json(d:Dict) -> str:
    return json.dumps(sorted(d.items()))

def from_json_to_dict(s:str) -> Dict:
    return json.loads(s) 

def from_pd_series_to_json(s:pd.Series) -> str:
    return from_dict_to_json(s.to_dict())

def from_json_to_pd_series(s:str) -> pd.Series:
    return pd.Series(from_json_to_dict(s))
