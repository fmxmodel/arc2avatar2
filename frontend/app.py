"""Arc2Avatar — Web Frontend

Upload a photo → run the pipeline → view 3D result in browser.
Serves on port 8888 (RunPod default).
"""

import os
import sys
import uuid
import json
import threading
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_file

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "uploads")
app.config["OUTPUT_FOLDER"] = os.path.join(os.path.dirname(__file__), "static", "results")
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

# In-memory job tracker
jobs: dict = {}


def run_pipeline_job(job_id: str, photo_path: str):
    """Run the Arc2Avatar pipeline in a background thread."""
    try:
        jobs[job_id]["status"] = "running"

        # 1. Validate input
        os.environ["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        from src.config.schema import load_and_validate_config, PipelineConfig
        from src.data.api import validate_input_image, extract_identity_embedding
        
        jobs[job_id]["progress"] = "Validating photo..."
        result = validate_input_image(photo_path)
        jobs[job_id]["face_info"] = result

        # 2. Load config (fast_debug for now)
        cfg = load_and_validate_config(experiment="fast_debug")
        cfg.data_prep.input_image_path = photo_path
        cfg.data_prep.id_embedding_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_embedding.npy")
        cfg.data_prep.id_embedding_json_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_embedding.json")
        cfg.export.ply_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_result.ply")
        cfg.export.turntable_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_turntable.mp4")
        cfg.run_id = job_id

        # 3. Run data prep
        jobs[job_id]["progress"] = "Extracting identity embedding..."
        extract_identity_embedding(photo_path, cfg.data_prep.id_embedding_path, cfg.data_prep.id_embedding_json_path)

        # 4. Generate a synthetic PLY for demo (or use real one)
        # In full mode, this would run the actual pipeline
        jobs[job_id]["progress"] = "Generating 3D result..."
        import numpy as np
        
        N = 5000
        means = np.random.randn(N, 3) * 0.5
        scales = np.ones((N, 3)) * 0.01
        rotations = np.zeros((N, 4))
        rotations[:, 0] = 1.0
        opacities = np.ones((N, 1)) * 0.8
        colors = np.random.rand(N, 3) * 0.5 + 0.5

        with open(cfg.export.ply_path, "w") as f:
            f.write(f"""ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
property float opacity
property float f_dc_0
property float f_dc_1
property float f_dc_2
end_header
""")
            for i in range(N):
                f.write(f"{means[i,0]:.6f} {means[i,1]:.6f} {means[i,2]:.6f} "
                        f"{scales[i,0]:.6f} {scales[i,1]:.6f} {scales[i,2]:.6f} "
                        f"{rotations[i,0]:.6f} {rotations[i,1]:.6f} {rotations[i,2]:.6f} {rotations[i,3]:.6f} "
                        f"{opacities[i,0]:.6f} "
                        f"{colors[i,0]:.6f} {colors[i,1]:.6f} {colors[i,2]:.6f}\n")

        jobs[job_id]["status"] = "completed"
        jobs[job_id]["progress"] = "Done!"
        jobs[job_id]["ply_url"] = f"/static/results/{job_id}_result.ply"

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["progress"] = str(e)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        return jsonify({"error": "No photo uploaded"}), 400

    file = request.files["photo"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    job_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or ".jpg"
    photo_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{job_id}{ext}")
    file.save(photo_path)

    jobs[job_id] = {
        "status": "queued",
        "progress": "Queued...",
        "face_info": None,
        "ply_url": None,
    }

    thread = threading.Thread(target=run_pipeline_job, args=(job_id, photo_path), daemon=True)
    thread.start()

    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    job = jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/results/<job_id>/ply")
def get_ply(job_id):
    ply_path = os.path.join(app.config["OUTPUT_FOLDER"], f"{job_id}_result.ply")
    if not os.path.exists(ply_path):
        return jsonify({"error": "PLY not found"}), 404
    return send_file(ply_path, mimetype="application/octet-stream")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8888))
    print(f"Arc2Avatar Web Frontend → http://0.0.0.0:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
