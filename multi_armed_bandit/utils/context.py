import json
import pandas as pd

from typing import Dict

def from_dict_to_str(d:Dict) -> str:
    return json.dumps(sorted(d.items()))

def from_str_to_dict(s:str) -> Dict:
    raise NotImplementedError

def from_pd_series_to_str(s:pd.Series) -> str:
    return from_dict_to_str(s.to_dict())
