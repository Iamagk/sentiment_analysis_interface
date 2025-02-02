from typing import Dict, Any
from rfsent import run_random_forest_model
from xgbsent import run_xgboost_model
from lgbsent import run_lightgbm_model

def run_model(model_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if model_name == "random_forest":
        return run_random_forest_model(data)
    elif model_name == "xgboost":
        return run_xgboost_model(data)
    elif model_name == "lightgbm":
        return run_lightgbm_model(data)
    else:
        raise ValueError(f"Unknown model: {model_name}")
