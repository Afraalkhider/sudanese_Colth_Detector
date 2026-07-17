import os
import json
import time
from datetime import datetime
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

model = YOLO('best.pt')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULTS_FILE = 'detection_results.json'


def save_result_to_json(image_name, detections):
    record = {
        'image': image_name,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'detections': detections
    }

    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = []

    all_results.append(record)

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')

    # In case the user uploads an image (POST request)
    if request.method == 'POST':
        uploaded_file = request.files['image']
        
        if uploaded_file.filename != '':
            # 1. إنشاء اسم مميز للصورة باستخدام الوقت عشان ما تتمسح
            timestamp = int(time.time())
            unique_filename = f"{timestamp}_{uploaded_file.filename}"
            original_path = os.path.join(UPLOAD_FOLDER, unique_filename)
            
            # حفظ الصورة الأصلية
            uploaded_file.save(original_path)

            # 2. إرسال الصورة للموديل
            results = model(original_path, conf=0.6)
            
            detected_items = []
            labels_data = [] # قائمة لحفظ الـ Labels
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                item_name = model.names[class_id]
                health_percent = round(float(box.conf[0]) * 100, 2)
                
                # 1. استخراج الإحداثيات أولاً (السطر ده كان ناقص)
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                # 2. تجهيز شكل الإحداثيات (في سطر واحد متصل)
                coords_text = f"Coords: [X1:{int(x1)}, Y1:{int(y1)}, X2:{int(x2)}, Y2:{int(y2)}]"
                
                # 3. إضافتها للنص اللي بيظهر في المتصفح (في سطر واحد متصل)
                detected_items.append(f"Detected: {item_name} | Health: {health_percent}% | {coords_text}")
                
                # 4. حفظها في ملف تجميع الـ Labels
                labels_data.append(f"Label: {item_name}, Confidence: {health_percent}%")

            # 3. حفظ ملف الـ Labels الخاص بالصورة دي
            # الملف حيكون بنفس اسم الصورة بس نهايته .txt
            label_filename = f"{timestamp}_labels.txt"
            label_path = os.path.join(UPLOAD_FOLDER, label_filename)
            
            with open(label_path, 'w', encoding='utf-8') as f:
                for label in labels_data:
                    f.write(label + '\n')

            # 4. حفظ الصورة بعد رسم المربعات عليها
            annotated_filename = 'detected_' + unique_filename
            annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
            results[0].save(filename=annotated_path)

            # 5. إرجاع النتيجة للواجهة
            return render_template('index.html', 
                                   image_url=annotated_path, 
                                   items=detected_items)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)