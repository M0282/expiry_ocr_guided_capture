from expiry_ocr import OCRLine, _extract_from_line, resolve_missing_year
from datetime import date


def test_expiry_korean():
    line = OCRLine("유통기한 26.09.13 22:15 까지", 92, "test")
    c = _extract_from_line(line)[0]
    assert c.kind == "expiry"
    assert c.year == 2026
    assert c.month == 9
    assert c.day == 13
    assert c.hour == 22
    assert c.minute == 15


def test_manufactured():
    line = OCRLine("제조일자 11.03 22:15", 90, "test")
    c = _extract_from_line(line)[0]
    assert c.kind == "manufactured"
    assert c.year is None
    assert c.month == 11
    assert c.day == 3


def test_missing_year_resolution():
    line = OCRLine("소비기한 11.14", 95, "test")
    c = _extract_from_line(line)[0]
    assert resolve_missing_year(c, date(2026, 9, 13)) == "2026-11-14"
