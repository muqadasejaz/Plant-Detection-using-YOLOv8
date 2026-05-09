## 🌿 Plant Detection using YOLOv8

A computer vision-based project that uses **YOLOv8** to detect and classify plant leaves in real time, helping identify healthy and diseased plants efficiently to support smart agriculture solutions.

<img width="1408" height="768" alt="Plant Diseases Detection App" src="https://github.com/user-attachments/assets/0f768603-ed60-4b66-80a2-bd10a1e1f6af" />


-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧾 Overview
This project focuses on **real-time plant disease detection** using the **YOLOv8 (You Only Look Once)** deep learning model.  
The system is designed to accurately detect and classify plant diseases from images and videos. It also includes tracking functionality to follow detected plants or leaves across video frames using `track.py`.

The goal of this project is to help farmers and researchers identify plant diseases early, ensuring better crop health and productivity.

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## 🌐 Live Demo

Click here to try the live app 👉 [Plant Diseases Detection App](https://plant-dectection.streamlit.app/)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


## ✨ Features
- 🌱 Real-time plant and leaf detection using YOLOv8  
- 🧠 Trained on a robust open-source dataset  
- 📊 High detection accuracy and confidence scoring  
- 🎥 Real-time object tracking using `track.py`  
- ☁️ Model training performed on **Google Colab**  
- 💾 Dataset stored and accessed via **Google Drive**  

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🧰 Tools and Technologies

| Category | Tools / Libraries |
|-----------|------------------|
| Programming Language | Python |
| Deep Learning Framework | PyTorch |
| Object Detection Model | YOLOv8 (Ultralytics) |
| Data Handling | Roboflow |
| Visualization | OpenCV, Matplotlib |
| Development Platform | Jupyter Notebook / Google Colab / Kaggle |

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📂 Project Structure

├── README.md 

├── YOLOv8_plant_detection.ipynb

├── apple.mp4  

├── requirements.txt 

└── track.py

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🚀 Usage

### 1. Clone the Repository
```bash
git clone https://github.com/muqadasejaz/-Plant-Detection-using-YOLOv8
```

### 2. Install Dependencies
```bash
pip install ultralytics opencv-python matplotlib
```

### 3. Dataset Setup

- Download the dataset from Roboflow Plant Disease Dataset

- Upload it to your Google Drive

- Mount Google Drive in your Colab notebook:
  ```bash
  from google.colab import drive
  drive.mount('/content/drive')
  ```

- Extract and link the dataset path in your code.

### 4. Train the Model

Open ```train.ipynb``` in Google Colab and run all cells.
After training, your model weights will be saved in the ```runs/train/weights/ directory```

### 5. Perform Detection

To test on new images or videos:
```bash
  yolo task=detect mode=predict model=runs/train/weights/best.pt source='path/to/image_or_video'
```

### 6. Tracking

Run the tracking script to follow detected plants in real-time videos:
```bash
python track.py
```

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📊 Dataset

The dataset used for this project is sourced from Roboflow Universe
.
It is titled “Plant Diseases Detection and Classification” and contains labeled images of various healthy and diseased plant leaves for multiple species.

### 🧠 Overview

Total Images: ~2,600

Image Type: RGB (JPEG/PNG)

Annotation Format: YOLOv8 (Bounding Boxes)

Split Ratio:

Train: 80%

Validation: 10%

Test: 10%

### 🌿 Classes

The dataset covers multiple types of plant diseases along with healthy leaves.
Below are some example classes:

Apple Scab

Apple Rust

Corn Leaf Blight

Corn Gray Spot

Potato Early Blight

Potato Late Blight

Tomato Bacterial Spot

Tomato Leaf Mold

Tomato Mosaic Virus

Healthy Leaf

### 🔗 Dataset Source

This dataset was prepared and published by contributors of **Graduation Project 2023** on **Roboflow Universe**.  
It was curated for research and experimentation in **plant disease detection and classification** using computer vision models such as **YOLOv8**.  

📎 **Dataset Link:** [Plant Diseases Detection and Classification – Roboflow Universe](https://universe.roboflow.com/graduation-project-2023/plants-diseases-detection-and-classification)  

📚 **Source Platform:** [Roboflow](https://roboflow.com/)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📈 Results  

The YOLOv8 model was trained and tested on the **Plant Diseases Detection and Classification** dataset from Roboflow.  
The results demonstrate that the model performs efficiently in detecting and classifying plant diseases across multiple species.  

### 🔬 Model Performance  
| Metric | Value |
|---------|--------|
| **Model** | YOLOv8n |
| **Epochs** | 50 |
| **Image Size** | 640x640 |
| **Precision** | 0.91 |
| **Recall** | 0.89 |
| **mAP@50** | 0.93 |
| **mAP@50-95** | 0.87 |

### 🌿 Detection Examples  
- **Healthy Leaf → 99% Confidence**  
- **Tomato Leaf Mold → 95% Confidence**  
- **Potato Early Blight → 92% Confidence**  
- **Apple Scab → 93% Confidence**

### 🖼️ Visualization  
The model accurately identifies diseased areas, drawing bounding boxes with class names and confidence scores on both images and videos.  

**Image Output:** 

![output](https://github.com/user-attachments/assets/330c3ba2-01b1-42bd-8a18-fae7874ac766)


![output1](https://github.com/user-attachments/assets/b1153a5a-677c-4263-8880-2211a7420a50)


![output3](https://github.com/user-attachments/assets/1253383f-072a-4092-88bc-bc09c5e21cbd)


![output4](https://github.com/user-attachments/assets/03bb798a-6e7f-47d9-8016-b9f51ba3b40e)


![output5](https://github.com/user-attachments/assets/e561b8b8-5598-4f9b-a424-5aeae0ec1b9e)


![output6](https://github.com/user-attachments/assets/e7ba57cd-a60f-427a-be54-be53511eac5a)

**Demo Output:**

The model was also tested on real-time plant videos. It successfully detected diseased regions frame-by-frame and displayed bounding boxes with corresponding class names and confidence scores.


https://github.com/user-attachments/assets/721ff69f-94c4-4e19-a1f6-907eab541693

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 🔖 References  

- [YOLOv8 by Ultralytics](https://github.com/ultralytics/ultralytics) — Official YOLOv8 repository and documentation.  
- [Roboflow: Plant Diseases Detection and Classification Dataset](https://universe.roboflow.com/graduation-project-2023/plants-diseases-detection-and-classification) — Dataset used for    model training and evaluation.  
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html) — Deep learning framework used to train and test the model.  
- [OpenCV Documentation](https://docs.opencv.org/) — Used for image preprocessing and visualization.  
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/) — Detailed usage guide and examples for YOLOv8.  

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 👤 Author

Muqadas Ejaz

BS Computer Science (AI Specialization)

AI/ML Engineer

Data Science & Gen AI Enthusiast

📫 Connect with me on [LinkedIn](https://www.linkedin.com/in/muqadasejaz/)  

🌐 GitHub: [github.com/muqadasejaz](https://github.com/muqadasejaz)

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## 📎 License

This project is open-source and available under the [MIT License](LICENSE).
