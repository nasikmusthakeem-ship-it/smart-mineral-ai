from flask import Flask, request, render_template_string
import os
import torch
from torchvision import transforms
from torchvision.models import resnet18
app = Flask(__name__)
from PIL import Image
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# Load trained mineral AI model
model = resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 4)

checkpoint = torch.load("mineral_model.pth", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

model.eval()

classes = checkpoint["classes"]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
print("MINERAL AI MODEL LOADED")
print("Classes:", classes)
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>SmartMiner AI</title>
    <style>
        body {
            font-family: Arial;
            background: #f4f7fb;
            text-align: center;
            padding: 50px;
        }

        .box {
            background: white;
            max-width: 650px;
            margin: auto;
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 5px 20px #ccc;
        }

        h1 {
            color: #182848;
        }

        h2 {
            color: #4b6cb7;
        }

        button {
            background: #4b6cb7;
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
        }

        input {
            margin: 20px;
        }
    </style>
</head>

<body>

<div class="box">

    <h1>🪨 SMART MINER AI</h1>

    <h2>AI-Assisted Mineral Sorting & Processing</h2>

    <p>Upload a mineral image for analysis.</p>

    <form action="/analysis" method="POST" enctype="multipart/form-data">

        <input type="file" name="mineral_image" accept="image/*" required>

        <br>

        <button type="submit">
            🔬 Analyze Mineral
        </button>

    </form>

</div>

</body>
</html>
"""

RESULT = """
<!DOCTYPE html>
<html>
<head>
    <title>SmartMiner AI - Result</title>
</head>

<body style="font-family:Arial;text-align:center;padding:50px;">

<h1>🪨 SMART MINER AI</h1>

<h2>Mineral Analysis Result</h2>

<p>Image uploaded successfully.</p>

<h3>🔬 AI Analysis Module</h3>

<p>Mineral identification system is ready.</p>

<p><b>Next stage:</b> AI model integration</p>

<a href="/">⬅ Back to Analysis</a>

</body>
</html>
"""

@app.route("/")
def home():
    return render_template_string(HTML)


@app.route("/analysis", methods=["POST"])
def analysis():

    image = request.files.get("mineral_image")

    if not image:
        return "No image uploaded."

    filepath = os.path.join(UPLOAD_FOLDER, image.filename)
    image.save(filepath)

    img = Image.open(filepath).convert("RGB")

    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    mineral = classes[predicted.item()]
    confidence_percent = confidence.item() * 100

    RESULT_AI = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>SmartMiner AI Result</title>
    </head>

    <body style="font-family:Arial;text-align:center;padding:50px;">

        <h1>🪨 SMART MINER AI</h1>

        <h2>Mineral Analysis Result</h2>

        <p><b>Predicted Mineral:</b></p>

        <h1 style="color:purple;">
            {mineral}
        </h1>

        <p>
            <b>AI Confidence:</b>
            {confidence_percent:.2f}%
        </p>

        <hr>

        <p>AI Model: ResNet18</p>

        <p>Classes: Ilmenite | Monazite | Rutile | Zircon</p>

        <br>

        <a href="/">← Analyze Another Image</a>

    </body>
    </html>
    """

    return render_template_string(RESULT_AI)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)