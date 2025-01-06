import gradio as gr
from ultralytics import YOLO
from PIL import Image

# Load YOLO model
model = YOLO("best2.pt")  

def predict(image):
    results = model(image)
    # Convert result to displayable image
    img_with_boxes = results[0].plot() 
    return Image.fromarray(img_with_boxes)

# Gradio interface
iface = gr.Interface(fn=predict, inputs="image", outputs="image", title="ASL Detector")

iface.launch(share=True)
