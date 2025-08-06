from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
import uuid
import glob
import time

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "inference/images/"
PROCESSED_FOLDER_ROOT = "runs/detect/"
MODEL_WEIGHTS = "./DataSetBurnIA.pt"
PROMPT_FILE = "final_prompt.txt"  # generado por detect.py en raíz
RESPONSE_FILE = "response_ia.txt"

@app.route("/processed/<path:filename>")
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER_ROOT, filename)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No se envió imagen'}), 400

    image = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # Ejecutar detect.py
    result = subprocess.run(
        ["python3", "detect.py", "--weights", MODEL_WEIGHTS, "--source", filepath],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        return jsonify({'error': 'Error al ejecutar el modelo', 'details': result.stderr}), 500

    time.sleep(1)  # esperar a que detect.py termine de escribir final_prompt.txt

    if not os.path.exists(PROMPT_FILE):
        return jsonify({'error': 'No se encontró final_prompt.txt'}), 500

    # Leer etiquetas y recomendación
    # Ejecutar response.sh y esperar a que termine correctamente
    response_script = "./response.sh"
    try:
        result = subprocess.run([response_script], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        return jsonify({
            "error": "Error al ejecutar response.sh",
            "details": e.stderr
        }), 500

    # Leer etiquetas y recomendación completas
    grados = []
    with open(PROMPT_FILE, "r", encoding="utf-8") as f:
        lineas = []
        for line in f:
            line = line.strip()
            lineas.append(line)
            if "Etiqueta detectada" in line:
                partes = line.split(":")[-1].strip().split()
                if len(partes) >= 3 and partes[1] == "degree":
                    grado = partes[0]
                    try:
                        confianza = float(partes[2])
                        grados.append((grado, confianza))
                    except ValueError:
                        continue


    recomendacion = ""
    with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
        lineas = []
        for line in f:
            line = line.strip()
            lineas.append(line)
        recomendacion = "\n".join(lineas)

    if not grados:
        return jsonify({
            "grado": "No detectado",
            "confianza": 0.0,
            "recomendaciones": recomendacion,
            "processedImage": None,
            "todasLasEtiquetas": []
        }), 200

    # Selecciona el grado más severo
    grados_orden = {"1st": 1, "2nd": 2, "3rd": 3}
    grado_detectado, confianza = max(grados, key=lambda x: grados_orden.get(x[0], 0))

    # Buscar imagen procesada
    exp_dirs = sorted(
        glob.glob(os.path.join(PROCESSED_FOLDER_ROOT, "exp*")),
        key=os.path.getmtime,
        reverse=True
    )
    latest_exp = exp_dirs[0]
    processed_path = os.path.join(latest_exp, filename)

    if not os.path.exists(processed_path):
        jpgs = glob.glob(os.path.join(latest_exp, "*.jpg"))
        if jpgs:
            processed_path = jpgs[0]
        else:
            processed_path = None

    processed_rel_path = (
        os.path.relpath(processed_path, PROCESSED_FOLDER_ROOT)
        if processed_path else None
    )

    return jsonify({
        "grado": f"{grado_detectado} degree",
        "confianza": round(confianza * 100, 1),
        "recomendaciones": recomendacion,
        "processedImage": f"/processed/{processed_rel_path.replace(os.sep, '/')}" if processed_rel_path else None,
        "todasLasEtiquetas": [
            {"grado": f"{g} degree", "confianza": round(c * 100, 1)} for g, c in grados
        ]
    })

if __name__ == "__main__":
#    app.run(port=5001, debug=True)
    app.run(host="10.0.20.51", port=5001, debug=True)

