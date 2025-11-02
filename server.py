from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import subprocess
import os
import uuid
import glob
import time

app = Flask(__name__)
CORS(app)

# --- Rutas absolutas y setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "inference", "images")
PROCESSED_FOLDER_ROOT = os.path.join(BASE_DIR, "runs", "detect")
MODEL_WEIGHTS = os.path.join(BASE_DIR, "DataSetBurnIA.pt")
PROMPT_FILE = os.path.join(BASE_DIR, "final_prompt.txt")   # generado por detect.py en raíz
RESPONSE_FILE = os.path.join(BASE_DIR, "response_ia.txt")
RESPONSE_SCRIPT = os.path.join(BASE_DIR, "response.sh")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER_ROOT, exist_ok=True)

def wait_for_file(path, timeout_sec=20, poll_ms=100):
    """Espera hasta que el archivo exista y tenga tamaño > 0, con timeout."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return True
        except FileNotFoundError:
            pass
        time.sleep(poll_ms / 1000.0)
    return False

@app.route("/processed/<path:filename>")
def serve_processed(filename):
    # Sirve desde runs/detect/ con rutas seguras
    return send_from_directory(PROCESSED_FOLDER_ROOT, filename)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({'ok': False, 'message': 'No se envió imagen'}), 400

    image = request.files['image']
    filename = f"{uuid.uuid4().hex}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    image.save(filepath)

    # Ejecutar detect.py (mantenemos flags actuales)
    try:
        result = subprocess.run(
            ["python3", os.path.join(BASE_DIR, "detect.py"), "--weights", MODEL_WEIGHTS, "--source", filepath],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        return jsonify({'ok': False, 'message': 'Error al ejecutar el modelo', 'details': e.stderr}), 500

    # --- Localizar imagen procesada en el exp* más reciente ---
    exp_dirs = sorted(
        glob.glob(os.path.join(PROCESSED_FOLDER_ROOT, "exp*")),
        key=os.path.getmtime,
        reverse=True
    )
    processed_path = None
    if exp_dirs:
        latest_exp = exp_dirs[0]
        candidate = os.path.join(latest_exp, filename)
        if os.path.exists(candidate):
            processed_path = candidate
        else:
            # Fallback: cualquier .jpg dentro del último exp
            jpgs = sorted(glob.glob(os.path.join(latest_exp, "*.jpg")))
            processed_path = jpgs[0] if jpgs else None

    processed_rel_path = (
        os.path.relpath(processed_path, PROCESSED_FOLDER_ROOT).replace(os.sep, "/")
        if processed_path else None
    )

    # --- Esperar archivos de texto (PROMPT y RESPONSE) en vez de sleep fijo ---
    _ = wait_for_file(PROMPT_FILE, timeout_sec=15)

    # Ejecutar response.sh y esperar su salida (si falla, seguimos sin recomendación)
    try:
        subprocess.run([RESPONSE_SCRIPT], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        pass

    # Leer PROMPT_FILE (si existe)
    prompt_lines = []
    if os.path.exists(PROMPT_FILE):
        with open(PROMPT_FILE, "r", encoding="utf-8") as f:
            prompt_lines = [ln.strip() for ln in f]

    # Leer RESPONSE_FILE (si existe)
    recomendacion = ""
    if wait_for_file(RESPONSE_FILE, timeout_sec=10):
        with open(RESPONSE_FILE, "r", encoding="utf-8") as f:
            recomendacion = "\n".join([ln.strip() for ln in f])

    # --- Parseo de grados (igual que el tuyo, con pequeño tolerance) ---
    grados = []
    for line in prompt_lines:
        if "degree" in line:
            partes = line.replace("Etiqueta detectada", "").replace(":", " ").split()
            grado = next((p for p in partes if p in {"1st", "2nd", "3rd"}), None)
            confianza = None
            for p in partes:
                try:
                    val = float(p)
                    confianza = val
                    break
                except ValueError:
                    continue
            if grado and confianza is not None:
                # 🔹 Cambia aquí: convierte 0.9 -> 90.0
                if confianza <= 1:
                    confianza = confianza * 100
                confianza = round(confianza, 1)
                grados.append((grado, confianza))

    # --- Respuesta coherente en ambos caminos (con o sin detecciones) ---
    if not grados:
        payload = {
            "ok": True,
            "message": "No se detectó quemadura",
            "grado": "No detectado",
            "confianza": 0.0,
            "recomendaciones": recomendacion,
            # AHORA sí enviamos processedImage si la tenemos
            "processedImage": f"/processed/{processed_rel_path}" if processed_rel_path else None,
            "todasLasEtiquetas": []
        }
        print("RESPUESTA /predict:", payload)
        return jsonify(payload), 200

    # Seleccionar grado más severo
    grados_orden = {"1st": 1, "2nd": 2, "3rd": 3}
    grado_detectado, confianza = max(grados, key=lambda x: grados_orden.get(x[0], 0))

    payload = {
        "ok": True,
        "message": "Detección realizada",
        "grado": f"{grado_detectado} degree",
        "confianza": round(confianza, 1),
        "recomendaciones": recomendacion,
        "processedImage": f"/processed/{processed_rel_path}" if processed_rel_path else None,
        "todasLasEtiquetas": [
            {"grado": f"{g} degree", "confianza": round(c, 1)} for g, c in grados
        ]
    }
    print("RESPUESTA /predict:", payload)
    return jsonify(payload), 200

if __name__ == "__main__":
    # app.run(port=5001, debug=True)
    app.run(host="10.0.20.50", port=5001, debug=True)
