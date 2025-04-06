from flask import Flask, render_template, request, jsonify, url_for, redirect, send_from_directory
from dotenv import load_dotenv
import os
from werkzeug.utils import secure_filename
from PIL import Image
import io
import time
import uuid
import base64  # Import base64
import json  # Import the json module

load_dotenv()
app = Flask(__name__)

UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploaded_images')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def save_images(files, tower_id):
    """Saves uploaded images to a directory named after the tower ID."""
    tower_folder = os.path.join(app.config['UPLOAD_FOLDER'], tower_id)
    os.makedirs(tower_folder, exist_ok=True)
    saved_filenames = []
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(tower_folder, filename)
            file.save(filepath)
            saved_filenames.append(filename)
    return saved_filenames, tower_folder

# --- Antenna Detection (YOLOv8) ---
def run_yolov8_detection(image_path):
    """
    Runs the YOLOv8 model on the given image path to detect antennas.
    """
    try:
        from ultralytics import YOLO

        model = YOLO('train3best.pt')
        results = model.predict(image_path)

        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                class_name = model.names[class_id]
                confidence = float(box.conf[0])
                if confidence > 0.5:
                    detections.append({
                        'class': class_name,
                        'bbox': [x1, y1, x2, y2],
                    })
        return detections

    except Exception as e:
        print(f"Error in run_yolov8_detection: {e}")
        return []

# --- Gemini Analysis ---
def analyze_with_gemini(image_data_list, bounding_boxes_list, prompt):
    """
    This is a placeholder for your actual Gemini API interaction.
    Now includes image data.
    """
    results = []
    for i, image_path in enumerate(image_data_list):  # Change to image_path
        try:
            # Read the image file as bytes
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
            # Encode the image bytes as a base64 string
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")

            # Include the image data in the prompt.  Adjust this based on how your API expects images.
            full_prompt_with_image = prompt + f"\nImage {i+1} (Base64 Encoded):\n{image_base64}"

            analysis_text = f"Simulated analysis for image {i+1} based on prompt:\n'{full_prompt_with_image[:200]}...\n" # limit to 200
            if bounding_boxes_list and len(bounding_boxes_list) > i and bounding_boxes_list[i]:
                analysis_text += f"Antennas detected: {bounding_boxes_list[i]}\n"
            analysis_text += "HEALTH:\n- N/A.\n- N/A.\n"
            analysis_text += "ANOMALIES:\n- N/A.\n- N/A.\n"
            analysis_text += "PERFORMANCE:\n- N/A.\n- N/A.\n"
            results.append({"analysis": analysis_text})
            time.sleep(1)
        except Exception as e:
            results.append({"error": str(e)})
    return results

STATIC_GEMINI_PROMPT = ("Analyze these images. For each tower, provide the following information:\n"
                        "TYPE OF TOWER: <monopole, lattice, or guyed>\n"
                        "HEALTH:\n"
                        "- <bullet point about health>\n"
                        "- <bullet point about health>\n"
                        "ANOMALIES:\n"
                        "- <bullet point about anomalies>\n"
                        "- <bullet point about anomalies>\n"
                        "PERFORMANCE:\n"
                        "- <bullet point about performance>\n"
                        "- <bullet point about performance>\n"
                        "Every new prompt will be a different tower, but images in the same prompt are the same tower from different angles. "
                        "If there isn't enough evidence to provide a bullet point, write \"N/A\" for that bullet point.")


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/insert.html', methods=['GET'])
def insert_page():
    return render_template('insert.html')

@app.route('/upload', methods=['POST'])
def upload_images_for_analysis():
    if 'images' not in request.files:
        return redirect(request.url)

    files = request.files.getlist('images')
    if not any(files):
        return "No files uploaded", 400

    tower_id = str(uuid.uuid4())
    uploaded_filenames, tower_folder = save_images(files, tower_id)

    image_data_list = [os.path.join(tower_folder, filename) for filename in uploaded_filenames] #list of image paths
    antenna_detections = []

    for image_path in image_data_list:
        detections = run_yolov8_detection(image_path)
        antenna_detections.append(detections)

    # Format antenna detection results for Gemini prompt
    antenna_info_prompt = ""
    for i, detections in enumerate(antenna_detections):
        antenna_info_prompt += f"Image {i+1} detected antennas:\n"
        if detections:
            for detection in detections:
                antenna_info_prompt += f"- Type: {detection['class']}, Bounding Box: {detection['bbox']}\n"
        else:
            antenna_info_prompt += "- No antennas detected.\n"

    full_gemini_prompt = STATIC_GEMINI_PROMPT + "\n" + antenna_info_prompt
    bounding_boxes_list = [
        [detection['bbox'] for detection in detections] if detections else []
        for detections in antenna_detections
    ]
    analysis_results = analyze_with_gemini(image_data_list, bounding_boxes_list, full_gemini_prompt) # Pass image paths

    # Convert analysis results to JSON
    analysis_results_json = json.dumps(analysis_results)

    return render_template('analysis_results.html', filenames=uploaded_filenames, analysis_results=analysis_results_json, tower_id=tower_id)

@app.route('/gallery.html')
def gallery():
    """Displays a list of towers, showing one representative image per tower."""
    towers = []
    tower_dirs = [d for d in os.listdir(app.config['UPLOAD_FOLDER']) if os.path.isdir(os.path.join(app.config['UPLOAD_FOLDER'], d))]
    for tower_id in tower_dirs:
        tower_path = os.path.join(app.config['UPLOAD_FOLDER'], tower_id)
        image_files = [f for f in os.listdir(tower_path) if allowed_file(f)]
        if image_files:
            # Choose the first image as the representative image
            representative_image = image_files[0]
            towers.append({
                'id': tower_id,
                'representative_image': representative_image,
                'image_count': len(image_files)
            })
    # Convert towers data to JSON
    towers_json = json.dumps(towers)
    return render_template('gallery.html', towers=towers_json)

@app.route('/profile/<tower_id>')
def view_profile(tower_id):
    """Displays all images and analysis results for a specific tower."""
    tower_path = os.path.join(app.config['UPLOAD_FOLDER'], tower_id)
    if not os.path.isdir(tower_path):
        return "Tower not found", 404
    image_files = [f for f in os.listdir(tower_path) if allowed_file(f)]

    # Get the analysis results.
    analysis_results = []
    # analysis_file_path = os.path.join(tower_path, "analysis.json")
    # if os.path.exists(analysis_file_path):
    #     with open(analysis_file_path, 'r') as f:
    #         analysis_results = json.load(f)
    analysis_results_json = json.dumps(analysis_results) #convert to json

    return render_template('profile.html', tower_id=tower_id, images=image_files, analysis_results=analysis_results_json)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return url_for('static', filename=filename)

@app.route('/uploads/<tower_id>/<filename>')
def serve_image(tower_id, filename):
    """Serves images from the tower-specific upload folder."""
    return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER'], tower_id), filename=filename)

if __name__ == '__main__':
    app.run(debug=True)