# 유통기한/소비기한 OCR 프로토타입

식품 포장지 사진에서 `유통기한`, `소비기한`, `제조일자`와 날짜/시간을 추출하는 독립 기능입니다.
나중에 모바일 앱에서 HTTP API로 호출할 수 있도록 `FastAPI`까지 포함했습니다.

## 처리 순서

1. 사진 입력
2. OpenCV로 확대/대비 강화/이진화 등 여러 전처리 버전 생성
3. Tesseract OCR을 여러 전처리 버전에 반복 적용
4. `유통기한`, `소비기한`, `제조일자`, `EXP`, `MFG` 등의 라벨 판별
5. 날짜 정규식 추출
6. OCR confidence + 라벨 존재 여부로 후보 점수화
7. 가장 가능성 높은 유통/소비기한 반환

## Windows 설치

Python 패키지:

```bash
pip install -r requirements.txt
```

Tesseract 본체도 설치해야 합니다. 설치 후 `tesseract --version`이 터미널에서 실행되어야 합니다.
가능하면 한국어 언어 데이터(`kor`)도 설치합니다.

## CLI 실행

```bash
python expiry_ocr.py sample.jpg
```

전처리 결과까지 확인하려면:

```bash
python expiry_ocr.py sample.jpg --debug debug_out
```

## API 실행

```bash
uvicorn api:app --reload --port 8000
```

브라우저:

```text
http://127.0.0.1:8000/docs
```

`POST /extract-expiry`에 사진을 업로드하면 JSON으로 결과가 나옵니다.

예시:

```json
{
  "expiry": {
    "kind": "expiry",
    "raw": "11.14",
    "year": null,
    "month": 11,
    "day": 14,
    "hour": 22,
    "minute": 15,
    "score": 0.86,
    "source_text": "유통기한 11.14 22:15",
    "predicted_iso_date": "2026-11-14"
  }
}
```

## 앱에 붙일 때 권장 UX

카메라 촬영 직후 OCR 결과를 곧바로 확정하지 말고:

`소비기한 2026-11-14 맞나요?  [확인] [수정]`

형태로 사용자에게 한 번 확인받는 것이 안전합니다.

특히 포장지에는 연도가 생략된 `11.14` 형식이 흔하므로,
자동으로 연도를 정한 값은 `predicted_iso_date`로만 전달하고 사용자가 최종 확인하도록 하는 것을 권장합니다.

## 다음 단계

Tesseract가 도트 매트릭스 인쇄에서 정확도가 부족하면 OCR 엔진만 PaddleOCR로 교체할 수 있도록
현재 구조를 유지한 채 `ocr_engine.py` 인터페이스를 분리하면 됩니다.
실제 앱에서는 촬영 가이드 박스로 날짜 인쇄 영역만 잘라 보내면 정확도가 크게 좋아집니다.


# 가이드 촬영 프로토타입

이번 버전에는 `guided_capture.py`가 추가되어 있습니다.

## 실행

```powershell
python .\guided_capture.py
```

카메라 창이 뜨면:

1. 유통기한 인쇄 부분만 중앙 가이드 박스 안에 맞춥니다.
2. 흔들림/밝기/대비가 기준을 만족하면 빨간 박스가 초록색으로 바뀝니다.
3. `SPACE`를 누르면 품질을 다시 검사합니다.
4. 품질 불량이면 촬영하지 않고 재촬영을 요구합니다.
5. 통과하면 박스 내부만 OCR합니다.
6. `Q` 또는 `ESC`로 종료합니다.

디버그 결과는 `guided_debug` 폴더에 저장됩니다.

현재는 PC 웹캠용 기능 검증 프로토타입입니다. 나중에 모바일 앱에서는 동일한 구조로
카메라 프리뷰 위에 가이드 박스를 표시하고, 박스 안의 ROI만 서버/온디바이스 OCR로 넘기면 됩니다.

### 품질 기준 튜닝

`guided_capture.py` 상단의 값을 조절할 수 있습니다.

```python
BLUR_THRESHOLD = 105.0
MIN_BRIGHTNESS = 55.0
MAX_BRIGHTNESS = 225.0
MIN_CONTRAST = 22.0
```

웹캠 특성에 따라 너무 자주 재촬영이 뜨면 `BLUR_THRESHOLD`를 70~90 정도로 낮추고,
너무 흐린 사진도 통과하면 130~180 정도로 올려보세요.
