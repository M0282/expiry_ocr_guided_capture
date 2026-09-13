from pathlib import Path
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from expiry_ocr import extract_expiry_info, resolve_missing_year, DateCandidate

app = FastAPI(title="Expiry OCR Prototype", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로토타입. 실제 앱 배포 시 도메인 제한 권장.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/extract-expiry")
async def extract_expiry(file: UploadFile = File(...)):
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드할 수 있습니다.")

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / f"input{suffix}"
        with path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        try:
            result = extract_expiry_info(path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

        # 연도 없는 날짜는 앱에서 확인용 predicted_iso_date도 같이 제공
        if result.get("expiry"):
            c = DateCandidate(**result["expiry"])
            result["expiry"]["predicted_iso_date"] = resolve_missing_year(c)

        return result
