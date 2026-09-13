from __future__ import annotations

import re
import time
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont


# =========================
# 튜닝 가능한 품질 기준
# =========================
BLUR_THRESHOLD = 105.0
MIN_BRIGHTNESS = 55.0
MAX_BRIGHTNESS = 225.0
MIN_CONTRAST = 22.0
GOOD_FRAMES_REQUIRED = 10

# 가이드 박스: 화면 중앙 기준
ROI_WIDTH_RATIO = 0.76
ROI_HEIGHT_RATIO = 0.25


DATE_PATTERNS = [
    # 2026.09.13 / 2026-09-13 / 2026/09/13
    re.compile(r"(?<!\d)(20\d{2})[.\-/](0?[1-9]|1[0-2])[.\-/](0?[1-9]|[12]\d|3[01])(?!\d)"),
    # 26.09.13
    re.compile(r"(?<!\d)(\d{2})[.\-/](0?[1-9]|1[0-2])[.\-/](0?[1-9]|[12]\d|3[01])(?!\d)"),
    # 09.13
    re.compile(r"(?<!\d)(0?[1-9]|1[0-2])[.\-/](0?[1-9]|[12]\d|3[01])(?!\d)"),
]

TIME_PATTERN = re.compile(r"(?<!\d)([01]?\d|2[0-3])[:](\d{2})(?!\d)")


def find_tesseract():
    """
    PATH에 tesseract가 없더라도 Windows 기본 설치 위치를 자동 확인한다.
    """
    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for p in candidates:
        if p.exists():
            pytesseract.pytesseract.tesseract_cmd = str(p)
            return str(p)
    return None


def get_font(size: int):
    candidates = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\malgunsl.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
    ]
    for p in candidates:
        if Path(p).exists():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def draw_text(frame, text, xy, size=26, color=(255, 255, 255)):
    """
    OpenCV putText는 한글 표시가 깨질 수 있어서 PIL로 한글을 그린다.
    color는 RGB.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil)
    draw.text(xy, text, font=get_font(size), fill=color)
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def roi_rect(frame):
    h, w = frame.shape[:2]
    rw = int(w * ROI_WIDTH_RATIO)
    rh = int(h * ROI_HEIGHT_RATIO)
    x1 = (w - rw) // 2
    y1 = (h - rh) // 2
    return x1, y1, x1 + rw, y1 + rh


def quality_metrics(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # 흔들림/초점: Laplacian variance
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # 밝기/대비
    brightness = float(gray.mean())
    contrast = float(gray.std())

    return {
        "blur_score": blur_score,
        "brightness": brightness,
        "contrast": contrast,
    }


def quality_message(metrics):
    if metrics["blur_score"] < BLUR_THRESHOLD:
        return False, f"흔들림/초점 불량 ({metrics['blur_score']:.0f}) - 카메라를 고정해주세요."
    if metrics["brightness"] < MIN_BRIGHTNESS:
        return False, f"너무 어둡습니다 ({metrics['brightness']:.0f}) - 조명을 밝게 해주세요."
    if metrics["brightness"] > MAX_BRIGHTNESS:
        return False, f"너무 밝습니다 ({metrics['brightness']:.0f}) - 빛 반사를 줄여주세요."
    if metrics["contrast"] < MIN_CONTRAST:
        return False, f"글자 대비가 약합니다 ({metrics['contrast']:.0f}) - 더 가까이 촬영해주세요."
    return True, "촬영 가능 - 유통기한 글자가 박스 안에 들어왔는지 확인하세요."


def normalize_digit_text(text: str) -> str:
    text = text.upper()
    text = (
        text.replace("O", "0")
            .replace("Q", "0")
            .replace("I", "1")
            .replace("L", "1")
            .replace("|", "1")
            .replace("：", ":")
            .replace("．", ".")
            .replace("。", ".")
            .replace("·", ".")
            .replace(",", ".")
            .replace("／", "/")
            .replace("－", "-")
    )
    text = re.sub(r"\s+", "", text)
    # 날짜 OCR에서 흔한 연속 구두점 정리
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r":{2,}", ":", text)
    return text


def preprocess_variants(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    scale = 3
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    contrast = clahe.apply(up)

    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharp = cv2.addWeighted(contrast, 1.8, blur, -0.8, 0)

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    adaptive = cv2.adaptiveThreshold(
        sharp, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )

    return {
        "gray": up,
        "contrast": contrast,
        "sharp": sharp,
        "otsu": otsu,
        "adaptive": adaptive,
    }


def ocr_roi(roi):
    variants = preprocess_variants(roi)

    configs = [
        "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789.:-/ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "--oem 3 --psm 7 -c tessedit_char_whitelist=0123456789.:-/ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        "--oem 3 --psm 11 -c tessedit_char_whitelist=0123456789.:-/ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    ]

    raw_results = []
    candidates = []

    for vname, img in variants.items():
        for cfg in configs:
            text = pytesseract.image_to_string(img, lang="eng", config=cfg)
            normalized = normalize_digit_text(text)
            if normalized:
                raw_results.append({
                    "variant": vname,
                    "text": text.strip(),
                    "normalized": normalized,
                })

                for idx, pattern in enumerate(DATE_PATTERNS):
                    for m in pattern.finditer(normalized):
                        raw = m.group(0)
                        if idx == 0:
                            year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
                        elif idx == 1:
                            year, month, day = 2000 + int(m.group(1)), int(m.group(2)), int(m.group(3))
                        else:
                            year, month, day = None, int(m.group(1)), int(m.group(2))

                        # 같은 문자열 안의 시간 후보
                        tm = TIME_PATTERN.search(normalized)
                        hour = int(tm.group(1)) if tm else None
                        minute = int(tm.group(2)) if tm else None

                        # 긴 날짜 형식을 우선시
                        score = {0: 1.0, 1: 0.9, 2: 0.75}[idx]
                        candidates.append({
                            "raw": raw,
                            "year": year,
                            "month": month,
                            "day": day,
                            "hour": hour,
                            "minute": minute,
                            "score": score,
                            "source": normalized,
                            "variant": vname,
                        })

    # 동일 날짜 후보가 여러 전처리에서 반복 검출되면 보너스
    counts = {}
    for c in candidates:
        key = (c["year"], c["month"], c["day"])
        counts[key] = counts.get(key, 0) + 1

    for c in candidates:
        key = (c["year"], c["month"], c["day"])
        c["score"] = round(min(1.0, c["score"] + 0.03 * (counts[key] - 1)), 3)

    # 중복 제거
    dedup = {}
    for c in candidates:
        key = (c["year"], c["month"], c["day"], c["hour"], c["minute"])
        if key not in dedup or c["score"] > dedup[key]["score"]:
            dedup[key] = c

    final_candidates = sorted(dedup.values(), key=lambda x: x["score"], reverse=True)

    return {
        "best": final_candidates[0] if final_candidates else None,
        "candidates": final_candidates[:10],
        "raw_ocr": raw_results[:20],
    }


def save_debug(roi, folder="guided_debug"):
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out / "captured_roi.jpg"), roi)
    for name, img in preprocess_variants(roi).items():
        cv2.imwrite(str(out / f"{name}.png"), img)


def main():
    find_tesseract()

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("카메라를 열 수 없습니다. 다른 앱이 카메라를 사용 중인지 확인해주세요.")

    good_frames = 0
    last_msg = ""
    flash_until = 0.0

    print("카메라 창이 열립니다.")
    print("유통기한 인쇄부를 가이드 박스 안에 맞춘 뒤 SPACE를 누르세요.")
    print("Q 또는 ESC: 종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        x1, y1, x2, y2 = roi_rect(frame)
        roi = frame[y1:y2, x1:x2]

        metrics = quality_metrics(roi)
        good, msg = quality_message(metrics)

        if good:
            good_frames = min(GOOD_FRAMES_REQUIRED, good_frames + 1)
        else:
            good_frames = 0

        ready = good_frames >= GOOD_FRAMES_REQUIRED

        # BGR: ready=green, otherwise red
        box_color = (0, 200, 0) if ready else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 3)

        title = "유통기한 인쇄 부분을 박스 안에 맞춰주세요"
        frame = draw_text(frame, title, (30, 25), 28, (255, 255, 255))

        status = "READY - SPACE를 눌러 촬영하세요" if ready else msg
        status_rgb = (80, 255, 80) if ready else (255, 100, 100)
        frame = draw_text(frame, status, (30, frame.shape[0] - 75), 24, status_rgb)

        metric_text = (
            f"sharpness {metrics['blur_score']:.0f} | "
            f"brightness {metrics['brightness']:.0f} | "
            f"contrast {metrics['contrast']:.0f}"
        )
        cv2.putText(
            frame, metric_text,
            (30, frame.shape[0] - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65,
            (255, 255, 255), 2, cv2.LINE_AA
        )

        if time.time() < flash_until:
            frame = draw_text(frame, last_msg, (30, 70), 25, (255, 220, 80))

        cv2.imshow("Expiry guided capture", frame)

        key = cv2.waitKey(1) & 0xFF

        if key in (27, ord("q"), ord("Q")):
            break

        if key == 32:  # SPACE
            metrics = quality_metrics(roi)
            good, msg = quality_message(metrics)

            if not good:
                last_msg = "재촬영 필요: " + msg
                flash_until = time.time() + 2.5
                print(last_msg)
                continue

            save_debug(roi)

            print("\n===== 촬영 품질 통과 =====")
            print(metrics)
            print("OCR 처리 중...")

            result = ocr_roi(roi)

            print("\n===== OCR RESULT =====")
            print(result)

            if result["best"]:
                best = result["best"]
                print("\n인식 후보:")
                print(best)
                last_msg = f"인식 성공: {best['raw']}"
            else:
                print("\n날짜를 찾지 못했습니다. 다시 촬영해주세요.")
                last_msg = "날짜 인식 실패 - 조금 더 가까이 다시 촬영해주세요"

            flash_until = time.time() + 3.0

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
