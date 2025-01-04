# SignBot Project

SignBot is an advanced system designed to interpret and recognize sign language gestures using computer vision and deep learning techniques. Leveraging the YOLO (You Only Look Once) architecture, the project achieves accurate and efficient gesture recognition, paving the way for better communication accessibility.

---

## Introduction

SignBot addresses the communication barriers faced by individuals who use sign language as their primary mode of interaction. By utilizing cutting-edge machine learning models, SignBot translates gestures into comprehensible outputs, making it a valuable tool for inclusivity.

---

## Features

- Real-time sign language gesture recognition
- High accuracy using YOLO architecture
- Easy-to-use interface
- Scalable for additional gestures and languages

---

## Tech Stack

- Real-time ASL sign recognition.
- Intuitive user interface for uploading images or live video streams.
- Accurate predictions using a fine-tuned YOLO model.
- Seamless hosting on Hugging Face for public access.

---

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/guptashreyas/SignBot.git
   cd signbot
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the pre-trained YOLO weights or train your own model.

4. Run the application:
   ```bash
     webcam.py
   ```

---

##Technology Stack

1.YOLOv11: For real-time object detection and ASL sign recognition.
2.Gradio: To build the interactive UI.
3.Hugging Face Spaces: For deployment and public hosting.
4.Python: Programming language for model integration and backend logic.

---

## Dataset

- Source: [Roboflow](https://public.roboflow.com/object-detection/american-sign-language-letters)
- Includes labeled images of various sign language gestures.
- Preprocessed for YOLO training (bounding boxes, augmentation).

---

## Usage

  1. Open the hosted SignBot on Hugging Face [SignBot](https://huggingface.co/spaces/shreyas001/ASL-Detector).
  2. Upload an image or use the webcam to capture a sign.
  3. View the predicted result in real time

## How It Works

  1.Model Training:
      -The YOLO model was trained on labeled ASL sign images using the Roboflow dataset.
      -Data augmentation techniques were applied for better generalization.
      
  2.User Interaction:
      -Users can upload an image or use their webcam to capture a live feed.
      -The YOLO model processes the input and predicts the recognized sign.

  3.Output:
      -The prediction result is displayed on the Gradio interface, along with confidence scores.
---

## Results

- Achieved high accuracy with YOLO for gesture recognition.
- Effective real-time predictions with minimal latency.

---

## License

This project is licensed under the [MIT License](LICENSE).

