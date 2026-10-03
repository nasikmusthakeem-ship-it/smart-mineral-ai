from flask import Flask, request, render_template_string
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

    if image:
        filepath = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(filepath)

    return render_template_string(RESULT)


if __name__ == "__main__":
    app.run(debug=True)