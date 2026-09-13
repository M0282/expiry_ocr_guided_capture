import requests

IMAGE = "sample.jpg"
with open(IMAGE, "rb") as f:
    r = requests.post(
        "http://127.0.0.1:8000/extract-expiry",
        files={"file": (IMAGE, f, "image/jpeg")},
        timeout=30,
    )

print(r.status_code)
print(r.json())
