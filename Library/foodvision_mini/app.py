
import gradio as gr
from model import create_effnetb2_model
from timeit import default_number as timer
import os
from typing import Tuple, Dict
import torch
import random
from PIL import Image

class_names=['pizza', 'sushi', 'steak']

effnetb2, effnet2_transforms=create_effnetb2_model(num_classes=3)

effnetb2.load_state_dict(torch.load(f='09_pretrained_effnetb2_feature_extractor _pizza_steak_sushi_20_percent.pth'), map_location=torch.device('cpu'))


def predict(img) -> Tuple[Dict, float]:
    """Transforms and performs a prediction on img and returns prediction and time taken.
    """
    
    start_time = timer()
    
    
    img = effnetb2_transforms(img).unsqueeze(0)
    
    
    effnetb2.eval()
    with torch.inference_mode():
        
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    
    
    pred_time = round(timer() - start_time, 5)
    
    
    return pred_labels_and_probs, pred_time


example_list=[['examples/'+example] for example in os.listdir('examples')]


import gradio as gr


title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."


demo = gr.Interface(fn=predict, 
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), 
                             gr.Number(label="Prediction time (s)")], 
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)


demo.launch(debug=False) 


