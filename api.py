import math
import cv2
import json
import os
import uuid
import urllib.request
import threading
import pymysql
import numpy as np
import face_recognition
import faiss
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Optional


load_dotenv()

SEAWEED_HOST = os.getenv("SEAWEED_HOST", "10.35.237.38")
SEAWEED_PORT = os.getenv("SEAWEED_PORT", "8888")

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "10.35.237.181"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", "root"),
    "database": os.getenv("DB_NAME", "trabajo_integrador_soa"),
    "charset": os.getenv("DB_CHARSET", "utf8mb4"),
}

FAISS_DIM = 128


class BuildProfileResponse(BaseModel):
    personId: str
    processedImages: int
    validEmbeddings: int
    rejectedImages: int


class FaceRecognitionResponse(BaseModel):
    personId: Optional[str] = None
    nombre: Optional[str] = None
    apellido: Optional[str] = None
    confidence: float


app = FastAPI(title="Face Recognition API")

# --- FAISS global state ---
_faiss_index = None
_faiss_id_to_person = {}
_faiss_next_id = 0
_faiss_lock = threading.Lock()


def _build_faiss_index(conn):
    global _faiss_index, _faiss_id_to_person, _faiss_next_id
    with conn.cursor() as cursor:
        cursor.execute("SELECT personId, vector FROM embeddings")
        rows = cursor.fetchall()

    if not rows:
        _faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(FAISS_DIM))
        _faiss_id_to_person = {}
        _faiss_next_id = 0
        return

    vectors = []
    mapping = {}
    for idx, (pid, vec_json) in enumerate(rows):
        vectors.append(np.array(json.loads(vec_json), dtype=np.float32))
        mapping[idx] = pid

    vectors_np = np.array(vectors, dtype=np.float32)
    ids_np = np.arange(len(vectors), dtype=np.int64)

    _faiss_index = faiss.IndexIDMap(faiss.IndexFlatL2(FAISS_DIM))
    _faiss_index.add_with_ids(vectors_np, ids_np)
    _faiss_id_to_person = mapping
    _faiss_next_id = len(vectors)


def _add_to_index(person_id: str, vector):
    global _faiss_next_id
    vec = np.array(vector, dtype=np.float32).reshape(1, -1)
    faiss_id = np.array([_faiss_next_id], dtype=np.int64)
    _faiss_index.add_with_ids(vec, faiss_id)
    _faiss_id_to_person[_faiss_next_id] = person_id
    _faiss_next_id += 1


def _search_index(query_encoding, k=1):
    query = np.array(query_encoding, dtype=np.float32).reshape(1, -1)
    distances, indices = _faiss_index.search(query, k)
    return distances[0], indices[0]


@app.on_event("startup")
def on_startup():
    conn = get_db()
    try:
        _build_faiss_index(conn)
    finally:
        conn.close()


# --- DB helpers ---

def get_db():
    return pymysql.connect(**DB_CONFIG)


def person_exists(conn, person_id: str) -> bool:
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT 1 FROM personas WHERE personId = %s", (person_id,)
        )
        return cursor.fetchone() is not None


def insert_embedding(conn, person_id: str, vector):
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    eid_hex = uuid.uuid4().hex
    with conn.cursor() as cursor:
        cursor.execute(
            "INSERT INTO embeddings (embeddingId, personId, vector, created_at) "
            "VALUES (X'{}', %s, %s, NOW())".format(eid_hex),
            (person_id, json.dumps(vector)),
        )
    conn.commit()


def get_person_info(conn, person_id: str):
    with conn.cursor() as cursor:
        cursor.execute(
            "SELECT nombre, apellido FROM personas WHERE personId = %s",
            (person_id,),
        )
        row = cursor.fetchone()
        if row:
            return {"nombre": row[0], "apellido": row[1]}
        return {"nombre": None, "apellido": None}


def upload_to_seaweed(person_id: str, filename: str, data: bytes):
    url = f"http://{SEAWEED_HOST}:{SEAWEED_PORT}/uploads/{person_id}/{filename}"
    req = urllib.request.Request(url, data=data, method="PUT")
    urllib.request.urlopen(req)


# --- Endpoints ---

@app.post("/embeddings", response_model=BuildProfileResponse)
async def build_profile(
    personId: str = Form(...),
    images: List[UploadFile] = File(...),
):
    person_id = personId.strip()
    if not person_id:
        raise HTTPException(status_code=400, detail="personId is required")
    if not images:
        raise HTTPException(status_code=400, detail="at least one image is required")

    try:
        conn = get_db()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"database connection failed: {e}")

    try:
        if not person_exists(conn, person_id):
            raise HTTPException(
                status_code=404,
                detail=f"personId '{person_id}' not found in database",
            )

        valid = 0
        rejected = 0

        for i, image_file in enumerate(images):
            try:
                contents = await image_file.read()
                if not contents:
                    rejected += 1
                    continue

                filename = image_file.filename or f"image_{i}.jpg"
                upload_to_seaweed(person_id, filename, contents)

                arr = np.frombuffer(contents, dtype=np.uint8)
                image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if image is None:
                    rejected += 1
                    continue
            except Exception:
                rejected += 1
                continue

            faces_locs = face_recognition.face_locations(image)
            if len(faces_locs) != 1:
                rejected += 1
                continue

            fr = face_recognition.face_encodings(
                image, known_face_locations=faces_locs
            )
            if fr:
                valid += 1
                enc = fr[0]
                insert_embedding(conn, person_id, enc)
                with _faiss_lock:
                    _add_to_index(person_id, enc)
            else:
                rejected += 1

        return BuildProfileResponse(
            personId=person_id,
            processedImages=len(images),
            validEmbeddings=valid,
            rejectedImages=rejected,
        )
    finally:
        conn.close()


@app.post("/recognition", response_model=FaceRecognitionResponse)
async def face_recognition_endpoint(
    image: UploadFile = File(...),
    threshold: float = Form(0.6),
):
    try:
        contents = await image.read()
        if not contents:
            raise HTTPException(status_code=400, detail="empty image")
        arr = np.frombuffer(contents, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="invalid image")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="failed to read image")

    faces_locs = face_recognition.face_locations(img)
    if len(faces_locs) != 1:
        detail = "no face detected" if not faces_locs else "more than one face detected"
        raise HTTPException(status_code=400, detail=detail)

    fr = face_recognition.face_encodings(img, known_face_locations=faces_locs)
    if not fr:
        raise HTTPException(status_code=400, detail="could not generate face encoding")

    query_encoding = fr[0]

    with _faiss_lock:
        if _faiss_index.ntotal == 0:
            return FaceRecognitionResponse(personId=None, confidence=0.0)

        distances, indices = _search_index(query_encoding)

    best_distance = math.sqrt(max(0.0, float(distances[0])))
    best_idx = int(indices[0])
    confidence = max(0.0, 1.0 - best_distance)

    if best_idx != -1 and confidence > threshold:
        with _faiss_lock:
            best_person_id = _faiss_id_to_person.get(best_idx)

        if best_person_id:
            conn = get_db()
            try:
                info = get_person_info(conn, best_person_id)
            finally:
                conn.close()

            return FaceRecognitionResponse(
                personId=best_person_id,
                nombre=info["nombre"],
                apellido=info["apellido"],
                confidence=round(confidence, 4),
            )

    return FaceRecognitionResponse(
        personId=None,
        confidence=round(confidence, 4),
    )
