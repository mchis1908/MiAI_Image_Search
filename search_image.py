import math
import os
import json
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from PIL import Image
import pickle
import numpy as np
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Khởi tạo model YOLO
model_detect = YOLO(r"C:\Users\ASUS\Desktop\project\MiAI_Image_Search\best.pt")

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

# Định nghĩa ảnh cần tìm kiếm
search_image_path = "testimage\\demo_image.jpg"

# Phát hiện đối tượng trong ảnh
results_detect = model_detect.predict(search_image_path)

# Đọc ảnh gốc
img = cv2.imread(search_image_path)

# Vòng lặp qua các hộp và cắt đối tượng
cropped_images = []
for box in results_detect[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped_img = img[y1:y2, x1:x2]
    cropped_images.append(cropped_img)

# Tải các vector và đường dẫn từ file
vectors = pickle.load(open("vectors.pkl", "rb"))
paths = pickle.load(open("paths.pkl", "rb"))

# Tạo output
results = []

# Trích xuất và tìm kiếm từng đối tượng đã cắt
for i, cropped_img in enumerate(cropped_images):
    cropped_pil_img = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
    search_vector = extract_vector(model, cropped_pil_img)
    
    # Tính khoảng cách từ search_vector đến tất cả các vector trong cơ sở dữ liệu
    distances = np.linalg.norm(vectors - search_vector, axis=1)
    
    # Sắp xếp và lấy ra K vector có khoảng cách ngắn nhất
    K = 3
    ids = np.argsort(distances)[:K]
    
    # Lưu kết quả
    nearest_images = [{"path": paths[id], "distance": float(distances[id])} for id in ids]
    results.append({"object": i+1, "results": nearest_images})

# Lưu kết quả vào file JSON
output_path = "search_results.json"
with open(output_path, "w") as json_file:
    json.dump(results, json_file, indent=4)

# Hiển thị kết quả cho từng đối tượng
for i, nearest_images in enumerate(results):
    print(f"Đối tượng {i+1}:")
    fig = plt.figure(figsize=(15, 5))  # Tăng kích thước figure để chứa được nhiều ảnh hơn
    axes = []
    rows = 1  # Số hàng
    cols = K  # Số cột
    
    for j, result in enumerate(nearest_images["results"]):
        path = result["path"]
        dist = result["distance"]
        draw_image = Image.open(path)
        axes.append(fig.add_subplot(rows, cols, j + 1))
        axes[-1].set_title(f"Dist: {dist:.2f}")
        plt.imshow(draw_image)
    
    fig.tight_layout()
    plt.show()

print(f"Kết quả đã được lưu vào {output_path}")
