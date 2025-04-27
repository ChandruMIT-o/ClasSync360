import os
import json
import cv2
import torch
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from pymongo import MongoClient
from PIL import Image
from facenet_pytorch import MTCNN
from deepface import DeepFace
import tempfile
import pandas as pd
from collections import defaultdict

# ——— CONFIG ———
MONGO_URI       = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME         = "attendance_db"
COLLECTION      = "attendance"
TIMETABLE_FP    = "timetable.json"
LABELS_CSV      = "labels.csv"
DB_PATH         = "db"
SIM_THRESH      = 0.5
OCCLUS_THRESH   = 0.85
MARGIN_RATIO    = 0.5
COUNT_THRESH    = 3   # threshold for final presence

# ——— INIT ———
app     = Flask(__name__)
client  = MongoClient(MONGO_URI)
col     = client[DB_NAME][COLLECTION]
mtcnn   = MTCNN(keep_all=True, device='cuda' if torch.cuda.is_available() else 'cpu')
labels  = pd.read_csv(LABELS_CSV)
filename_to_label = {
    os.path.splitext(r['imagename'])[0].lower(): r['label']
    for _, r in labels.iterrows()
}
students_all = labels['label'].unique().tolist()

def save_detected_face(img_array, label, prefix):
    """Save face image to detected_faces/{prefix}/ with timestamp."""
    save_dir = os.path.join("detected_faces", prefix)
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{timestamp}_{label}.jpg"
    cv2.imwrite(os.path.join(save_dir, filename), img_array)


with open(TIMETABLE_FP) as f:
    timetable = json.load(f)

# ——— HELPERS ———

def find_current_subject(classid, now=None):
    now = now or datetime.now()
    day = now.strftime("%A")
    sched = timetable.get(classid, {}).get(day, {})
    for subj, info in sched.items():
        start = datetime.strptime(info['start'], "%H:%M").time()
        end   = datetime.strptime(info['end'],   "%H:%M").time()
        if start <= now.time() <= end:
            return subj, start, end
    return None, None, None

def process_and_match(stream):
    img   = Image.open(stream).convert("RGB")
    orig  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    boxes, probs = mtcnn.detect(img)
    results = []

    if boxes is None:
        return results

    for box, p in zip(boxes, probs):
        if p < OCCLUS_THRESH:
            continue

        # crop + margin
        x1, y1, x2, y2 = map(int, box)
        w, h = x2-x1, y2-y1
        mw, mh = int(w*MARGIN_RATIO), int(h*MARGIN_RATIO)
        x1, y1 = max(0, x1-mw), max(0, y1-mh)
        x2, y2 = min(orig.shape[1], x2+mw), min(orig.shape[0], y2+mh)
        crop = orig[y1:y2, x1:x2]

        # temp file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, crop, [int(cv2.IMWRITE_JPEG_QUALITY),100])
            tmp_path = tmp.name

        label = "Unknown"
        try:
            df = DeepFace.find(
                img_path=tmp_path,
                db_path=DB_PATH,
                enforce_detection=False,
                distance_metric='cosine'
            )[0]
            if not df.empty:
                df['reg_id'] = df['identity'].apply(
                    lambda p: os.path.basename(os.path.dirname(p))
                )
                df['sim']    = 1.0 - df['distance']

                # ── WEIGHTED VOTING (sum of sims) ──
                scores = defaultdict(float)
                for _, r in df.iterrows():
                    scores[r['reg_id']] += r['sim']
                best_id = max(scores, key=scores.get)
                best_sim = df[df['reg_id']==best_id]['sim'].max()
                if best_sim >= SIM_THRESH:
                    label = filename_to_label.get(best_id.lower(), best_id)
        except Exception as e:
            app.logger.warning(f"DeepFace.find failed: {e}")

        # cleanup
        try: os.remove(tmp_path)
        except: pass

        save_detected_face(crop, label, prefix="attendance")


        results.append(label)

    return results
# ——— helper to finalize a period ─────────────────────────────
def finalize_period(classid, date_str, subj):
    """Finalize attendance for one subject: 
       - read counts, apply COUNT_THRESH, write final presence booleans
       - set ongoing flag to False once only"""
    counts_path = f"attendance_counts.{date_str}.{subj}"
    flag_path   = f"attendance_flags.{date_str}.{subj}.ongoing"

    # load the document once
    doc = col.find_one({"_id": classid}) or {}
    counts = (
        doc.get("attendance_counts", {})
           .get(date_str, {})
           .get(subj, {})
    )

    # build $set payload: presence bool + count for each student
    final_updates = {}
    for stud, cnt in counts.items():
        final_updates[f"attendance.{date_str}.{subj}.{stud}"] = {
            "presence": cnt >= COUNT_THRESH,
            "count":    cnt
        }
    # flip the ongoing flag
    final_updates[flag_path] = False

    col.update_one({"_id": classid}, {"$set": final_updates})


# ——— modified /attendance endpoint ────────────────────────────
@app.route("/attendance", methods=["POST"])
def attendance():
    classid = request.form.get("classid")
    if not classid:
        return jsonify(error="classid is required"), 400

    now      = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    subj, start_t, end_t = find_current_subject(classid, now)
    if not subj:
        return jsonify(error="No active period"), 400

    # ——— 1) ensure only ONE ongoing per date —————————
    # fetch existing flags for this class/date
    doc_flags = (
        col.find_one({"_id": classid}, {"attendance_flags":1}) 
        or {}
    ).get("attendance_flags", {}).get(date_str, {})

    for other_subj, info in doc_flags.items():
        if other_subj != subj and info.get("ongoing") is True:
            # finalize the other subject first
            finalize_period(classid, date_str, other_subj)

    # paths for THIS subject
    flag_path   = f"attendance_flags.{date_str}.{subj}.ongoing"
    counts_path = f"attendance_counts.{date_str}.{subj}"

    # reload doc for this subject’s flag
    doc = col.find_one({"_id": classid}) or {}

    # ——— 2) INIT PHASE: first time we see this subject ——
    # only if never initialized or flagged False
    subj_flags = doc.get("attendance_flags", {}).get(date_str, {})
    if subj not in subj_flags or subj_flags[subj].get("ongoing") is False:
        # zero out all counts & set ongoing=True
        init_counts = {f"{counts_path}.{s}": 0 for s in students_all}
        col.update_one(
            {"_id": classid},
            {"$set": {flag_path: True, **init_counts}},
            upsert=True
        )

    # ——— 3) FINALIZE PHASE: if time’s up ————————
    elif doc["attendance_flags"][date_str][subj]["ongoing"] and now.time() > end_t:
        finalize_period(classid, date_str, subj)
        return jsonify(status="finalized"), 200

    # ——— 4) INCREMENT PHASE: regular image uploads ————
    increments = {}
    if 'image' in request.files:
        labels = process_and_match(request.files['image'])
        for lbl in labels:
            if lbl in students_all:
                # atomic +1
                increments[f"{counts_path}.{lbl}"] = 1

    if increments:
        col.update_one({"_id": classid}, {"$inc": increments})

    return jsonify(status="counted", increments=increments), 200

# ——— TIMETABLE ENDPOINT ———

@app.route("/timetable", methods=["GET"])
def get_timetable():
    return send_from_directory(
        directory=os.getcwd(),
        path=TIMETABLE_FP,
        mimetype="application/json"
    )

@app.route("/engagement", methods=["POST"])
def engagement():
    classid = request.form.get("classid")
    if not classid:
        return jsonify(error="classid is required"), 400

    now       = datetime.now()
    date_str  = now.strftime("%Y-%m-%d")
    subj, _, _= find_current_subject(classid, now)
    if not subj:
        return jsonify(error="No active period"), 400

    # Base field path for this hour
    base = f"engagement.{date_str}.{subj}"

    # We'll collect pushes as: { "<base>.<student>": emotion }
    pushes = {}

    if 'image' in request.files:
        img_stream = request.files['image']
        img = Image.open(img_stream).convert("RGB")
        orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        boxes, probs = mtcnn.detect(img)

        if boxes is not None:
            for box, p in zip(boxes, probs):
                if p < OCCLUS_THRESH:
                    continue

                # --- Crop with margin ---
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1
                mw, mh = int(w*MARGIN_RATIO), int(h*MARGIN_RATIO)
                x1, y1 = max(0, x1-mw), max(0, y1-mh)
                x2 = min(orig.shape[1], x2+mw)
                y2 = min(orig.shape[0], y2+mh)
                face = orig[y1:y2, x1:x2]

                # --- Temp save ---
                with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                    cv2.imwrite(tmp.name, face, [int(cv2.IMWRITE_JPEG_QUALITY),100])
                    tmp_path = tmp.name

                # --- 1) Identity matching (weighted voting) ---
                student = None
                try:
                    df = DeepFace.find(
                        img_path=tmp_path,
                        db_path=DB_PATH,
                        enforce_detection=False,
                        distance_metric='cosine'
                    )[0]
                    if not df.empty:
                        df['reg_id'] = df['identity'].apply(
                            lambda p: os.path.basename(os.path.dirname(p))
                        )
                        df['sim'] = 1.0 - df['distance']
                        # sum sims
                        scores = defaultdict(float)
                        for _, r in df.iterrows():
                            scores[r['reg_id']] += r['sim']
                        best = max(scores, key=scores.get)
                        top_sim = df[df['reg_id']==best]['sim'].max()
                        if top_sim >= SIM_THRESH:
                            student = filename_to_label.get(best.lower(), best)
                except:
                    pass

                # --- 2) Emotion only ---
                emotion = None
                if student:
                    try:
                        analysis = DeepFace.analyze(
                            img_path=tmp_path,
                            actions=['emotion'],
                            enforce_detection=False
                        )[0]
                        emotion = analysis.get('dominant_emotion')
                    except:
                        pass

                # cleanup
                try: os.remove(tmp_path)
                except: pass

                # --- 3) Queue Mongo push ---
                if student and emotion:
                    pushes.setdefault(f"{base}.{student}", []).append(emotion)

    if pushes:
        # Build $push each with $each
        update_doc = {"$push": {}}
        for field, em_list in pushes.items():
            update_doc["$push"][field] = {"$each": em_list}
        col.update_one({"_id": classid}, update_doc, upsert=True)

    return jsonify(status="appended", pushes=pushes), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
