\# Sudanese Traditional Clothing Detector 🇸🇩



\## Overview

This is a Flask-based web application that utilizes a custom-trained YOLO object detection model. The system is designed to identify and localize Sudanese traditional garments (Toub, Jallabiya, Markoub) from user-uploaded images. 



\## Features

\- \*\*Image Upload:\*\* Users can upload images directly through a simple web interface.

\- \*\*AI Detection:\*\* Utilizes a YOLO model (`best.pt`) to draw bounding boxes around detected items.

\- \*\*Health Score (Confidence):\*\* Calculates and displays the confidence percentage for each detected item.

\- \*\*Privacy-Focused:\*\* Uploaded and processed images are stored locally in the `static/uploads` directory and are excluded from version control.



\## Project Structure

```text

my\_yolo\_project/

│

├── app.py                # Main Flask application and backend logic

├── best.pt               # Custom trained YOLO model

├── README.md             # Project documentation

├── .gitignore            # Git exclusion rules (ignores /venv, /uploads)

├── static/

│   └── uploads/          # Directory for saving uploaded \& annotated images

└── templates/

&#x20;   └── index.html        # Frontend web interface

