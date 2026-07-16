import os
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# Load the model
model = YOLO('best.pt')

# Create a folder to save uploaded and analyzed images if it doesn't exist
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    # In case of a normal user visit to the site (GET request)
    if request.method == 'GET':
        return render_template('index.html')

    # In case the user uploads an image (POST request)
    if request.method == 'POST':
        uploaded_file = request.files['image']
        
        if uploaded_file.filename != '':
            # 1. Save the original image
            original_path = os.path.join(UPLOAD_FOLDER, uploaded_file.filename)
            uploaded_file.save(original_path)

            # 2. Send the image to the model for analysis
            results = model(original_path, conf=0.6)
            
            # 3. Extract data (Class name and Health percentage)
            detected_items = []
            for box in results[0].boxes:
                # Extract the class name
                class_id = int(box.cls[0])
                item_name = model.names[class_id]
                
                # Extract the health (confidence) score and convert it to a percentage
                health_percent = round(float(box.conf[0]) * 100, 2)
                
                # Add the result to the list
                detected_items.append(f"Detected: {item_name} | Health: {health_percent}%")

            # 4. Save the new annotated image (with bounding boxes)
            annotated_filename = 'detected_' + uploaded_file.filename
            annotated_path = os.path.join(UPLOAD_FOLDER, annotated_filename)
            
            # YOLO library command to save the image
            results[0].save(filename=annotated_path)

            # 5. Return the webpage along with the analyzed image and data
            return render_template('index.html', 
                                   image_url=annotated_path, 
                                   items=detected_items)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)