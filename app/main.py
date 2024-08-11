from flask import Flask, request, jsonify, render_template_string
from inference_sdk import InferenceHTTPClient
import os

app = Flask(__name__)

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key=os.getenv("ROBOFLOW_API_KEY")
)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def home():
    html_content = '''
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Home</title>
        <style>
          body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            font-family: Arial, sans-serif;
            background-color: #f8f9fa;
          }
          .container {
            text-align: center;
          }
          h1 {
            color: #343a40;
          }
        </style>
      </head>
      <body>
        <div class="container">
          <h1>Server is up and running!</h1>
          <p>Visit <code>/predict</code> to use the prediction endpoint.</p>
        </div>
      </body>
    </html>
    '''
    return render_template_string(html_content)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
    try:
        image_file.save(image_path)
        result = CLIENT.infer(image_path, model_id="fashion-detector-n3bi3/3")
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
