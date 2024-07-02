import os
import sys
import torch
import numpy as np
from fastapi import FastAPI
from typing import List
from pydantic import BaseModel

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from config import device
from utils.pretreatment_image import pretreatment_image
from src.models.convnet import ConvNet
from src.models.mlp import MLP

app = FastAPI()

# def predict(model, image):
#     model.to(device)
#     model.eval()

#     with torch.no_grad():
#         y_pred = model(image)
    
#     return torch.argmax(y_pred, dim=1).item()

def predict(model, input: np.ndarray) -> np.ndarray:
    """
    Run model and get result
    :param package: dict from fastapi state including model and processing objects
    :param input: list of input values or an image in a suitable format
    :return: numpy array of model output
    """

    # Preprocess the data if necessary
    # input = preprocess(input)
    image = pretreatment_image(input)

    # Run the model
    # model = package['model']
    model.to(device)
    model.eval()

    with torch.no_grad():
        y_pred = model(image)

    pred = torch.argmax(y_pred, dim=1)

    # Convert result to a numpy array on CPU
    # pred = y_pred.cpu().detach().numpy()
    # pred = pred.cpu().numpy()
    pred_list = pred.cpu().tolist()

    return pred_list



convnet = ConvNet(input_size=28*28, n_kernels=6, output_size=10)
convnet.load_state_dict(torch.load("model/convnet_model.pt", map_location=torch.device(device)))

mlp = MLP(input_size=28*28, n_hidden=8, output_size=10)
mlp.load_state_dict(torch.load("model/mlp_model.pt", map_location=torch.device(device)))

class InferenceInput(BaseModel):
    file: List[List[float]]

@app.post("/api/v1/predict")
async def predict_digit(body: InferenceInput):
    prediction = predict(convnet, np.array(body.file))
    print(f"Server side prediction : {prediction}")
    return {"prediction": prediction}
