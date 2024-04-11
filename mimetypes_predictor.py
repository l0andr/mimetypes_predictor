from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Dict, Any
from time import time
from datetime import date

@dataclass_json
@dataclass
class MIMETypePredictorModel:
    '''
    MIMETypePredictorModel is a dataclass that represents the model for MIME Type prediction.
    '''
    date: str
    supported_types: List[str]
    number_of_train_samples: int
    model_data: Dict[str, Any]
    model_version: float
    f1_score: float
    def __str__(self):
        return f"Date: {self.date}, Supported Types: {self.supported_types}, Number of Train Samples: {self.number_of_train_samples}\n" \
               f"Model Version: {self.model_version}, F1 Score: {self.f1_score}"


class MIMETypePredictor:
    '''
    MIMETypePredictor is a class that predicts the MIME Type of a given data.
    '''
    model_version = 0.1
    def __init__(self, model: MIMETypePredictorModel):
        if model.model_version != self.model_version:
            raise ValueError(f"Model version mismatch. Expected {self.model_version}, got {model.model_version}")
        self.model = model

    def train(self, data: List[str]) -> MIMETypePredictorModel:
        raise NotImplementedError

    def predict(self, data: Any) -> str:
        raise NotImplementedError




if __name__ == '__main__':
    pass