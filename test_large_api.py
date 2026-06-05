"""Test the API with a large image similar to what a phone camera would produce."""
import requests, base64, json, numpy as np, cv2

# Create a realistic-size image (like a phone photo)
img = np.random.randint(100, 200, (1920, 1080, 3), dtype=np.uint8)
_, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
b64 = base64.b64encode(buf).decode()
data_uri = "data:image/jpeg;base64," + b64

payload = {"image": data_uri}
payload_str = json.dumps(payload)
print(f"Image size: {img.shape}")
print(f"JPEG size: {len(buf)} bytes ({len(buf)/1024:.0f} KB)")
print(f"Base64 size: {len(b64)} bytes ({len(b64)/1024:.0f} KB)")
print(f"Total payload: {len(payload_str)} bytes ({len(payload_str)/1024/1024:.1f} MB)")

print("\nSending to API...")
try:
    resp = requests.post(
        "http://127.0.0.1:5000/api/process_image",
        json=payload,
        timeout=120
    )
    print(f"Status: {resp.status_code}")
    data = resp.json()
    print(f"Success: {data.get('success')}")
    if data.get('error'):
        print(f"Error: {data['error']}")
    if data.get('timing'):
        print(f"Processing time: {data['timing'].get('total', '?')}s")
except Exception as e:
    print(f"Request failed: {e}")
