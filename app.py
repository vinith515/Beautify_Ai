import os
from dotenv import load_dotenv
load_dotenv()  # loads REPLICATE_API_TOKEN from .env file

import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import traceback

from inference.pipeline import AcneRemovalPipeline

current_dir = os.path.dirname(os.path.abspath(__file__))
dist_folder = os.path.join(current_dir, 'beautifyai modified', 'beautifyai', 'beautifyai', 'dist')

app = Flask(__name__, static_folder=dist_folder)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)

print("[INFO] Loading pipeline...")
pipe = AcneRemovalPipeline(smooth_strength=0.8)
print("[INFO] Ready!")


@app.route('/api/process_image', methods=['POST'])
def process_image():
    try:
        data = request.get_json(force=True, silent=True)
        if not data or 'image' not in data:
            return jsonify({'success': False, 'error': 'No image provided'}), 400

        base64_img = data['image']
        if ',' in base64_img:
            base64_img = base64_img.split(',')[1]

        img_data = base64.b64decode(base64_img)
        np_arr = np.frombuffer(img_data, np.uint8)
        img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({'success': False, 'error': 'Invalid image data'}), 400

        print(f"[INFO] Image: {img_bgr.shape}, processing...")
        result = pipe.process(img_bgr)
        final_bgr = result['output']

        _, buffer = cv2.imencode('.jpg', final_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
        processed_base64 = base64.b64encode(buffer).decode('utf-8')

        total_t = result.get('timing', {}).get('total', 0)
        print(f"[INFO] Done. Time: {total_t:.2f}s")

        return jsonify({
            'success': True,
            'image': f"data:image/jpeg;base64,{processed_base64}",
            'timing': result.get('timing', {}),
            'identity_similarity': result.get('identity_similarity', 1.0),
            'spots_detected': result.get('spots_detected', 0),
        })

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
