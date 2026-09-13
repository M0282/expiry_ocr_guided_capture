from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Optional, Iterable

import cv2
import numpy as np
import pytesseract


DATE_LABELS = {
    "expiry": ("유통기한", "소비기한", "품질유지기한", "까지", "EXP", "EXPIRY", "BEST BEFORE", "USE BY"),
    "manufactured": ("제조일자", "제조", "생산일자", "MFG", "MFD", "MANUFACTURED"),
}

# 2026.09.13 / 26.09.13 / 09.13 / 11/14 / 11-14
DATE_RE = re.compile(
    r"(?<!\d)"
    r"(?:(?P<year>20\d{2}|\d{2})\s*[.\-/년]\s*)?"
    r"(?P<month>0?[1-9]|1[0-2])\s*[.\-/월]\s*"
    r"(?P<day>0?[1-9]|[12]\d|3[01])"
    r"(?:\s*일)?"
    r"(?!\d)"
)

# 22:15 / 13.20 (날짜와 혼동하지 않도록 별도 처리)
TIME_RE = re.compile(r"(?<!\d)(?P<hour>[01]?\d|2[0-3])\s*[:]\s*(?P<minute>[0-5]\d)(?!\d)")


@dataclass
class OCRLine:
    text: str
    confidence: float
    variant: str


@dataclass
class DateCandidate:
    kind: str
    raw: str
    year: Optional[int]
    month: int
    day: int
    hour: Optional[int]
    minute: Optional[int]
    score: float
    source_text: str

    def to_dict(self):
        return asdict(self)


def _normalize_ocr_text(text: str) -> str:
    # 날짜 숫자 주변에서 자주 생기는 OCR 혼동만 가볍게 보정.
    # 영문 라벨 자체를 망가뜨리지 않도록 전체적인 O->0 치환은 하지 않는다.
    text = text.replace("．", ".").replace("。", ".").replace("·", ".")
    text = text.replace("：", ":").replace("／", "/").replace("－", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _preprocess_variants(image: np.ndarray) -> dict[str, np.ndarray]:
    if image is None or image.size == 0:
        raise ValueError("빈 이미지입니다.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

    # 작은 도트 프린트 문자를 확대
    h, w = gray.shape[:2]
    scale = 3 if max(h, w) < 1400 else 2
    up = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 국부 대비 강화
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    contrast = clahe.apply(up)

    # 배경 노이즈 억제
    denoise = cv2.bilateralFilter(contrast, 7, 45, 45)

    # 샤프닝
    blur = cv2.GaussianBlur(denoise, (0, 0), 1.2)
    sharp = cv2.addWeighted(denoise, 1.8, blur, -0.8, 0)

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        sharp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )

    # 도트 프린트의 끊긴 점을 아주 약하게 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    close = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel, iterations=1)

    return {
        "gray": up,
        "contrast": contrast,
        "sharp": sharp,
        "otsu": otsu,
        "adaptive": adaptive,
        "close": close,
    }


def _ocr_variant(img: np.ndarray, variant: str) -> list[OCRLine]:
    """
    Tesseract TSV를 사용해 줄 단위 텍스트와 평균 confidence를 얻는다.
    한국어+영어 데이터가 설치되어 있으면 kor+eng, 아니면 eng로 자동 폴백한다.
    """
    try:
        langs = set(pytesseract.get_languages(config=""))
        lang = "kor+eng" if "kor" in langs and "eng" in langs else "eng"
    except Exception:
        lang = "eng"

    data = pytesseract.image_to_data(
        img,
        lang=lang,
        config="--oem 3 --psm 11",
        output_type=pytesseract.Output.DICT,
    )

    grouped = {}
    n = len(data["text"])
    for i in range(n):
        txt = _normalize_ocr_text(data["text"][i])
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except Exception:
            conf = -1.0

        key = (
            data.get("block_num", [0] * n)[i],
            data.get("par_num", [0] * n)[i],
            data.get("line_num", [0] * n)[i],
        )
        grouped.setdefault(key, []).append((txt, conf))

    lines = []
    for parts in grouped.values():
        text = " ".join(p[0] for p in parts)
        good_conf = [p[1] for p in parts if p[1] >= 0]
        conf = sum(good_conf) / len(good_conf) if good_conf else 0.0
        lines.append(OCRLine(text=text, confidence=conf, variant=variant))
    return lines


def _detect_kind(text: str) -> str:
    upper = text.upper()
    for kind, labels in DATE_LABELS.items():
        if any(label.upper() in upper for label in labels):
            return kind
    return "unknown"


def _parse_year(token: Optional[str]) -> Optional[int]:
    if not token:
        return None
    y = int(token)
    if y < 100:
        return 2000 + y
    return y


def _valid_calendar_date(year: Optional[int], month: int, day: int) -> bool:
    # 연도가 없으면 2024(윤년) 기준으로 월/일만 검증
    check_year = year or 2024
    try:
        date(check_year, month, day)
        return True
    except ValueError:
        return False


def _candidate_score(kind: str, ocr_conf: float, has_year: bool, has_time: bool) -> float:
    score = max(0.0, min(1.0, ocr_conf / 100.0)) * 0.45
    if kind != "unknown":
        score += 0.30
    if has_year:
        score += 0.15
    if has_time:
        score += 0.05
    # 날짜 패턴 자체가 유효한 경우 기본점수
    score += 0.05
    return round(min(score, 1.0), 3)


def _extract_from_line(line: OCRLine) -> list[DateCandidate]:
    text = _normalize_ocr_text(line.text)
    kind = _detect_kind(text)
    out = []

    times = list(TIME_RE.finditer(text))

    for m in DATE_RE.finditer(text):
        year = _parse_year(m.group("year"))
        month = int(m.group("month"))
        day = int(m.group("day"))

        if not _valid_calendar_date(year, month, day):
            continue

        hour = minute = None
        if times:
            # 같은 줄의 날짜 뒤에 가장 가까운 시간을 연결
            time_m = min(times, key=lambda t: abs(t.start() - m.end()))
            hour = int(time_m.group("hour"))
            minute = int(time_m.group("minute"))

        out.append(
            DateCandidate(
                kind=kind,
                raw=m.group(0),
                year=year,
                month=month,
                day=day,
                hour=hour,
                minute=minute,
                score=_candidate_score(kind, line.confidence, year is not None, hour is not None),
                source_text=text,
            )
        )
    return out


def _dedupe(candidates: Iterable[DateCandidate]) -> list[DateCandidate]:
    best = {}
    for c in candidates:
        key = (c.kind, c.year, c.month, c.day, c.hour, c.minute)
        if key not in best or c.score > best[key].score:
            best[key] = c
    return sorted(best.values(), key=lambda x: x.score, reverse=True)


def resolve_missing_year(candidate: DateCandidate, scan_date: Optional[date] = None) -> Optional[str]:
    """
    연도가 없는 MM.DD 표기를 앱에서 실제 날짜로 써야 할 때 선택적으로 사용.
    현재 연도 날짜가 scan_date보다 30일 이상 과거라면 다음 해로 간주한다.
    자동 확정이 싫다면 이 함수를 호출하지 말고 year=None 상태로 사용자 확인 UI를 띄우는 것이 안전하다.
    """
    scan_date = scan_date or date.today()
    year = candidate.year

    if year is None:
        y = scan_date.year
        try:
            d = date(y, candidate.month, candidate.day)
        except ValueError:
            return None

        if (scan_date - d).days > 30:
            y += 1
        year = y

    try:
        d = date(year, candidate.month, candidate.day)
        return d.isoformat()
    except ValueError:
        return None


def extract_expiry_info(image_path: str | Path, save_debug_dir: Optional[str | Path] = None) -> dict:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {image_path}")

    variants = _preprocess_variants(image)

    if save_debug_dir:
        debug_dir = Path(save_debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)
        for name, img in variants.items():
            cv2.imwrite(str(debug_dir / f"{name}.png"), img)

    all_lines = []
    for name, img in variants.items():
        try:
            all_lines.extend(_ocr_variant(img, name))
        except pytesseract.TesseractError:
            continue

    # 후보 생성
    candidates = []
    for line in all_lines:
        candidates.extend(_extract_from_line(line))

    # 라벨과 날짜가 OCR에서 서로 다른 줄로 갈라지는 경우 보완:
    # 라벨 줄 다음 후보에 kind를 전파하기 위해 전체 텍스트도 한 번 조합해서 분석한다.
    joined = "\n".join(l.text for l in all_lines)
    joined_kind = _detect_kind(joined)

    candidates = _dedupe(candidates)

    # expiry 후보가 전혀 없고 전체 OCR에 유통/소비기한 라벨이 존재하면
    # 가장 점수가 높은 unknown 후보를 expiry로 승격
    if joined_kind == "expiry" and not any(c.kind == "expiry" for c in candidates):
        for c in candidates:
            if c.kind == "unknown":
                c.kind = "expiry"
                c.score = round(min(c.score + 0.20, 1.0), 3)
                break

    expiry = next((c for c in candidates if c.kind == "expiry"), None)
    manufactured = next((c for c in candidates if c.kind == "manufactured"), None)

    # 라벨 인식이 실패했을 때는 최고점 후보를 generic best_date로만 반환한다.
    best = candidates[0] if candidates else None

    return {
        "image": str(image_path),
        "expiry": expiry.to_dict() if expiry else None,
        "manufactured": manufactured.to_dict() if manufactured else None,
        "best_date": best.to_dict() if best else None,
        "candidates": [c.to_dict() for c in candidates[:10]],
        "ocr_lines": [
            {"text": l.text, "confidence": round(l.confidence, 1), "variant": l.variant}
            for l in sorted(all_lines, key=lambda x: x.confidence, reverse=True)[:30]
        ],
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="식품 포장 유통/소비기한 OCR 프로토타입")
    parser.add_argument("image", help="입력 이미지 경로")
    parser.add_argument("--debug", default=None, help="전처리 이미지 저장 폴더")
    args = parser.parse_args()

    result = extract_expiry_info(args.image, args.debug)
    print(json.dumps(result, ensure_ascii=False, indent=2))
