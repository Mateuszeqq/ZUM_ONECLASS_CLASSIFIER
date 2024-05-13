from ucimlrepo import fetch_ucirepo
from src.constants import TARGET_CASTING


def get_data(id: int):
    data = fetch_ucirepo(id=id) 
   
    X = data.data.features

    y = data.data.targets['Class']
    
    if y.dtype != 'int64':
        y = data.data.targets.replace(TARGET_CASTING[id])

    X["target"] = y.values

    return X