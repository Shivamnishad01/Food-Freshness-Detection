from flask import Flask, render_template, request, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os

app = Flask(__name__)  # Flask will automatically use templates/ and static/ folders

# ==== 0. Upload folder (inside static) ====
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)   # Create folder if it doesn't exist

# ==== 1. Load Model ====
MODEL_PATH = "freshness_model.h5"   # Model file in the same directory
model = tf.keras.models.load_model(MODEL_PATH)

# Class names used during training
class_names = ["fresh", "rotten"]


# ==== 2. Home Route ====
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    img_url = None   # This will be passed to the frontend

    if request.method == "POST":
        file = request.files.get("file")   # The form input must have name="file"

        if file and file.filename != "":
            # Fixed filename for the uploaded image
            filename = "uploaded_preview.jpg"

            # Full server path for saving the uploaded file
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)

            # ===== Prepare Image for Model =====
            img = Image.open(img_path).convert("RGB")
            img = img.resize((224, 224))
            arr = np.array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)

            preds = model.predict(arr)
            probs = preds[0]
            max_prob = float(np.max(probs))
            idx = int(np.argmax(probs))
            label = class_names[idx]

            # Confidence threshold to detect non-food images
            threshold = 0.7
            if max_prob < threshold:
                prediction = "This image does not appear to be a food item (Model is unsure) 🤔"
            else:
                if label == "fresh":
                    prediction = f"Food is FRESH ✅ (confidence: {max_prob:.2f})"
                else:
                    prediction = f"Food is BAD / SPOILED ❌ (confidence: {max_prob:.2f})"

            # ==== Generate Image URL for Frontend Preview ====
            img_url = url_for("static", filename=f"uploads/{filename}")

    return render_template("mini.html", prediction=prediction, img_url=img_url)


if __name__ == "__main__":
    app.run(debug=True)