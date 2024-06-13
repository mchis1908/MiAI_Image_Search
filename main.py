from flask import Flask, request, jsonify
import requests
import numpy as np
import cv2
from PIL import Image
import io
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from ultralytics import YOLO
import matplotlib.pyplot as plt
import pickle

app = Flask(__name__)

# Khởi tạo model YOLO
model_detect = YOLO(r"model\best.pt")

# Hàm tạo model trích xuất đặc trưng
def get_extract_model():
    vgg16_model = VGG16(weights="imagenet")
    extract_model = Model(inputs=vgg16_model.inputs, outputs=vgg16_model.get_layer("fc1").output)
    return extract_model

# Hàm tiền xử lý ảnh
def image_preprocess(img):
    img = img.resize((224, 224))
    img = img.convert("RGB")
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Hàm trích xuất vector đặc trưng từ ảnh
def extract_vector(model, img):
    img_tensor = image_preprocess(img)
    vector = model.predict(img_tensor)[0]
    vector = vector / np.linalg.norm(vector)
    return vector

# Khởi tạo model trích xuất đặc trưng
model = get_extract_model()

@app.route('/api/search_image', methods=['POST'])
def search_image():
    data = request.get_json()
    image_url = data['image_url']

    # Tải ảnh từ URL
    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content))

    # Chuyển đổi ảnh từ PIL sang định dạng OpenCV
    img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Phát hiện đối tượng trong ảnh
    results_detect = model_detect.predict(source=img_cv2)

    # Vòng lặp qua các hộp và cắt đối tượng
    cropped_images = []
    for box in results_detect[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cropped_img = img_cv2[y1:y2, x1:x2]
        cropped_images.append(cropped_img)

    # Kiểm tra xem đã cắt được đối tượng nào chưa
    if not cropped_images:
        return jsonify({"error": "Không tìm thấy đối tượng nào trong ảnh."})

    # Tải các vector và đường dẫn từ file
    try:
        vectors = pickle.load(open("vectors.pkl", "rb"))
        paths = pickle.load(open("paths.pkl", "rb"))
    except FileNotFoundError as e:
        return jsonify({"error": f"Lỗi: {e}"})

    # Tạo output
    results = []

    # Trích xuất và tìm kiếm từng đối tượng đã cắt
    for i, cropped_img in enumerate(cropped_images):
        cropped_pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
        search_vector = extract_vector(model, cropped_pil_img)
        
        # Tính khoảng cách từ search_vector đến tất cả các vector trong cơ sở dữ liệu
        distances = np.linalg.norm(vectors - search_vector, axis=1)
        
        # Sắp xếp và lấy ra K vector có khoảng cách ngắn nhất
        valid_ids = np.where(distances <= 1.25)[0]
        
        if len(valid_ids) == 0:
            continue 

        K = 1
        ids = valid_ids[np.argsort(distances[valid_ids])][:K]
        
        # Lưu kết quả
        nearest_images = [{"path": paths[id], "distance": float(distances[id])} for id in ids]
        results.append({"object": i + 1, "results": nearest_images})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
