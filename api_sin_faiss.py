import cv2
import json
import os
import uuid
import urllib.request
import pymysql
import numpy as np
import face_recognition
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


def load_all_embeddings(conn):
    with conn.cursor() as cursor:
        cursor.execute("SELECT personId, vector FROM embeddings")
        return [
            {"personId": pid, "encoding": np.array(json.loads(vec))}
            for pid, vec in cursor.fetchall()
        ]


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

    try:
        conn = get_db()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"database connection failed: {e}")

    try:
        known = load_all_embeddings(conn)
        if not known:
            return FaceRecognitionResponse(personId=None, confidence=0.0)

        known_encodings = [item["encoding"] for item in known]
        distances = face_recognition.face_distance(known_encodings, query_encoding)
        best_idx = int(np.argmin(distances))
        best_distance = float(distances[best_idx])
        confidence = max(0.0, 1.0 - best_distance)

        if confidence > threshold:
            best_person_id = known[best_idx]["personId"]
            info = get_person_info(conn, best_person_id)
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
    finally:
        conn.close()
