from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Dict, Any,Optional
from time import time
from datetime import date
import numpy as np
import os
import pickle
import json
import pandas as pd
import tqdm
import argparse
from sklearn.metrics import classification_report, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
@dataclass_json
@dataclass
class MIMETypePredictorModel:
    '''
    MIMETypePredictorModel is a dataclass that represents the model for MIME Type prediction.
    '''
    date: str
    supported_types: List[str]
    number_of_train_samples: int
    model_data: List[int]
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
    def __init__(self, model: Optional[MIMETypePredictorModel]=None):
        if model is not None and model.model_version != self.model_version:
            raise ValueError(f"Model version mismatch. Expected {self.model_version}, got {model.model_version}")
        self.model = model
        self.train_vectors = []
        self.train_labels = []

    def extract_features(self, data: bytes) -> List[float]:
        #extract first 128 bytes and convert to float -> 128 byte features
        first128 = [float(x) for x in data[:128]]
        #compute vector of 256 elements with freqency of each byte -> 256 float features
        freqbytes = [0.0]*256
        for byte in data:
            freqbytes[byte] += 1.0
        freqbytes = [x/len(data) for x in freqbytes]
        #compute matrix of 256x256 of frequency of each pair of bytes -> compute singular values -> 256 float features
        freqpairs = [[0]*256 for _ in range(256)]
        for i in range(1,len(data)):
            freqpairs[data[i-1]][data[i]] += 1
        freqpairs = [[x/len(data) for x in y] for y in freqpairs]
        svd = np.linalg.svd(freqpairs,compute_uv=False)
        return first128 + freqbytes + list(svd)


    def train(self, data: List[bytes],labels:List[str],verbose:int=0) -> MIMETypePredictorModel:
        for i in tqdm.tqdm(range(len(data)),desc="Extracting features",disable=verbose<1):
            self.train_vectors.append(self.extract_features(data[i]))
            self.train_labels.append(labels[i])
        train_test_split_ratio = 0.3
        if train_test_split_ratio > 0.0:
            Xtrain, Xtest, ytrain, ytest = train_test_split(self.train_vectors, self.train_labels, test_size=train_test_split_ratio,
                                                            random_state=1)
        else:
            Xtrain = self.train_vectors
            ytrain = self.train_labels
            Xtest = Xtrain
            ytest = ytrain
        self.model_kernel = RandomForestClassifier()
        # split train and test
        self.model_kernel.fit(Xtrain, ytrain)
        accuracy = self.model_kernel.score(Xtest,ytest)
        in_sample_accuracy = self.model_kernel.score(Xtrain,ytrain)
        f1_score_value = f1_score(ytest,self.model_kernel.predict(Xtest),average='weighted')
        if verbose > 1:
            print(f"Model accuracy is: test {accuracy}, in-sample {in_sample_accuracy}. F1 score: {f1_score_value}")
        if verbose > 2:
            y_pred_test = self.model_kernel.predict(Xtest)
            print(classification_report(ytest, y_pred_test))
        model_byte_str = pickle.dumps(self.model_kernel)
        self.model = MIMETypePredictorModel(date=str(date.today()),supported_types=list(set(labels)),
                                            number_of_train_samples=len(data),model_data=model_byte_str,
                                            model_version=self.model_version,f1_score=f1_score_value)
    def save(self, path:str) -> None:
        json_str = self.model.to_json()
        with open(path,"w") as f:
            f.write(json_str)

    def load(self, path:str) -> None:
        json_data = ""
        with open(path,"rt") as f:
            json_data = f.read()
        self.model = MIMETypePredictorModel.from_json(json_data)
        self.model_kernel = pickle.loads(bytes(self.model.model_data))

    def model_accuracy(self,data: List[bytes],labels:List[str],verbose:int=0) -> float:
        vectors = []
        for i in tqdm.tqdm(range(len(data)),desc="Extracting features",disable=verbose<1):
            vectors.append(self.extract_features(data[i]))
        return self.model_kernel.score(vectors,labels)

    def model_f1_score(self,data: List[bytes],labels:List[str],verbose:int=0) -> float:
        vectors = []
        for i in tqdm.tqdm(range(len(data)),desc="Extracting features",disable=verbose<1):
            vectors.append(self.extract_features(data[i]))
        return f1_score(labels,self.model_kernel.predict(vectors),average='weighted')
    def predict(self, data: bytes) -> str:
        features = np.array(self.extract_features(data))
        return self.model_kernel.predict(features.reshape(1,len(features)))

def prepare_train_dataset(path:str,verbose:int=0) -> (List[bytes],List[str]):
    data = []
    labels = []
    for file in tqdm.tqdm(os.listdir(path),desc="Read train files",disable= (verbose<1)):
        with open(os.path.join(path,file),"rb") as f:
            data.append(f.read())
            labels.append(file.split(".")[-1])
    return data,labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and MIME Type predictor model",
        formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("-input_dir", help="Directory with files for model train", type=str, default="")
    parser.add_argument("-model", help="File for/with model. If file exists model will be loaded and tested on train data", type=str, required=True)
    parser.add_argument("-i", help="input file",type=str, default="")
    parser.add_argument("-verbose",help="Verbose level",type=int,default = 1)

    args = parser.parse_args()
    verbose = args.verbose
    input_dir = args.input_dir
    model_file = args.model
    if len(args.input_dir) == 0 and len(args.i) == 0:
        raise Exception("Either input_dir or input file should be ser")
    if len(args.input_dir) > 0:
        data, labels = prepare_train_dataset(input_dir, verbose)
        if verbose > 1:
            print(f"Set of train lables {set(labels)}")
    if len(args.i) > 0:
        data = [open(args.i, "rb").read()]
        labels = ["unknown"]
    if os.path.exists(model_file):
        if verbose > 0:
            print(f"Model file {model_file} exists. Loading model")
        model = MIMETypePredictor()
        model.load(model_file)
        label = model.predict(data[0])
        if verbose > 0:
            print(f"Predicted label for file {args.i}: {label}")
        else:
            print(label)
    elif len(data) > 1:
        if verbose > 0:
            print(f"Model file {model_file} does not exist. Training model")
        model = MIMETypePredictor()
        model.train(data=data, labels=labels, verbose=verbose)
        model.save(model_file)
    else:
        if verbose > 0:
            print(f"Model file {model_file} does not exist and train data do not specified. Exit")
