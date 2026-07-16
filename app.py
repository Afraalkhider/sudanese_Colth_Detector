import os
import json
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

    if request.method == 'POST':
        uploaded_file = request.files['image']
        
        if uploaded_file.filename != '':
            original_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(original_path)

            results = model(original_path, conf=0.6)
            
            detected_items = []
            json_detections = []
            
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                item_name = model.names[class_id]
                
                health_percent = round(float(box.conf[0]) * 100, 2)
                
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                
                detected_items.append(f"Detected: {item_name} | Health: {health_percent}%")
                
                json_detections.append({
                    'name': item_name,
                    'confidence': health_percent,
                    'box': {
                        'x1': round(x1, 1),
                        'y1': round(y1, 1),
                        'x2': round(x2, 1),
                        'y2': round(y2, 1)
                    }
                })

            annotated_filename = 'detected_' + uploaded_file.filename
            annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
            
            results[0].save(filename=annotated_path)

            save_result_to_json(uploaded_file.filename, json_detections)

            return render_template('index.html', 
                                   image_url=annotated_path, 
                                   items=detected_items)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)