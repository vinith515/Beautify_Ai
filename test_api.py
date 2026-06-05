import requests, base64, json, numpy as np, cv2

# Create a small test image
img = np.ones((100, 100, 3), dtype=np.uint8) * 150
_, buf = cv2.imencode('.jpg', img)
b64 = base64.b64encode(buf).decode()
data_uri = "data:image/jpeg;base64," + b64

payload = {"image": data_uri}
payload_str = json.dumps(payload)
print("Payload size:", len(payload_str), "bytes")

resp = requests.post(
    "http://127.0.0.1:5000/api/process_image",
    json=payload,
    timeout=60
)
print("Status:", resp.status_code)
print("Response:", resp.text[:500])
