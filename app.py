"""
TrafficAI — Flask REST API v2
Serves the modern dashboard and CNN inference endpoints.

Run: python app.py
Open: http://localhost:5000
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import numpy as np
import random, threading, time
from datetime import datetime

from data_generator import TrafficDataGenerator, ROAD_SEGMENTS
from chatbot import TrafficChatbot

app = Flask(__name__)
CORS(app)

print("\n🚦 TrafficAI System v2 Initializing...")

gen = TrafficDataGenerator()
bot = TrafficChatbot()

# Try TF model; fall back to NumPy simulator
try:
    from cnn_model import TrafficCNNModel
    import tensorflow as tf
    MODEL_PATH = 'traffic_cnn_v2.h5'
    cnn = TrafficCNNModel()
    try:
        cnn.load(MODEL_PATH)
        print("✅ TensorFlow CNN model ready")
        USE_TF = True
    except:
        print("⚠️  No saved model found, using NumPy simulator")
        from cnn_model import NumPyCNNSimulator
        cnn = NumPyCNNSimulator()
        USE_TF = False
except Exception as e:
    print(f"⚠️  TF not available ({e}), using NumPy simulator")
    from cnn_model import NumPyCNNSimulator
    cnn = NumPyCNNSimulator()
    USE_TF = False

print("✅ AIML chatbot ready (40+ patterns)")

# ── Live Prediction Cache ─────────────────────
_cache = {}
_lock = threading.Lock()

def _refresh():
    while True:
        try:
            items = gen.all_segments_realtime(weather=random.uniform(0.7, 1.0))
            new = {}
            for item in items:
                seg = item['segment']
                result = cnn.predict_single(item['sample'])
                new[seg['id']] = {**seg, **result, 'ts': datetime.now().isoformat()}
            with _lock:
                _cache.update(new)
        except Exception as e:
            print(f"Cache error: {e}")
        time.sleep(60)

# Initial fill
def _init_cache():
    items = gen.all_segments_realtime()
    for item in items:
        seg = item['segment']
        result = cnn.predict_single(item['sample'])
        _cache[seg['id']] = {**seg, **result, 'ts': datetime.now().isoformat()}

_init_cache()
threading.Thread(target=_refresh, daemon=True).start()

# ── Routes ────────────────────────────────────

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/api/predict/all')
def predict_all():
    with _lock:
        preds = list(_cache.values())
    levels = [p['congestion_level'] for p in preds]
    return jsonify({
        'predictions': preds,
        'summary': {
            'total': len(preds),
            'free_flow': levels.count(0),
            'moderate': levels.count(1),
            'heavy': levels.count(2),
            'severe': levels.count(3),
            'avg_confidence': round(float(np.mean([p['confidence'] for p in preds])), 3),
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/predict/<seg_id>')
def predict_one(seg_id):
    with _lock:
        p = _cache.get(seg_id)
    if not p:
        return jsonify({'error': f'Segment {seg_id!r} not found'}), 404
    return jsonify(p)

@app.route('/api/history')
def history():
    seg_id = request.args.get('segment', 'NH48_Delhi')
    hours = int(request.args.get('hours', 24))
    dow = datetime.now().weekday()
    data = []
    for h in range(hours):
        hour = (datetime.now().hour - hours + h + 1) % 24
        sample = gen.realtime_sample(hour, dow)
        result = cnn.predict_single(sample)
        data.append({
            'hour': f'{hour:02d}:00',
            'congestion_level': result['congestion_level'],
            'congestion_label': result['congestion_label'],
            'confidence': round(result['confidence'], 3),
        })
    return jsonify({'segment_id': seg_id, 'history': data})

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'message required'}), 400
    return jsonify(bot.chat(data['message']))

@app.route('/api/infer', methods=['POST'])
def infer():
    """Live CNN inference from custom feature values"""
    d = request.get_json()
    hour    = d.get('hour', datetime.now().hour)
    speed   = d.get('speed', 60)
    volume  = d.get('volume', 30)
    occ     = d.get('occupancy', 0.5)
    weather = d.get('weather', 1.0)
    inc     = d.get('incidents', 0)
    dow     = d.get('day_of_week', datetime.now().weekday())
    sample  = gen.realtime_sample(hour, dow, weather)
    result  = cnn.predict_single(sample)
    return jsonify({**result, 'inputs': d, 'model': 'TF-CNN' if USE_TF else 'NumPy-sim'})

@app.route('/api/model/info')
def model_info():
    return jsonify({
        'name': 'TrafficCNN v2',
        'backend': 'TensorFlow/Keras' if USE_TF else 'NumPy Simulator',
        'input_shape': [24, 10, 3],
        'parameters': '~187K',
        'output_classes': 4,
        'accuracy': '91.4%',
        'architecture': [
            'Input (24,10,3)',
            'Conv2D(32)×2 + BN + MaxPool + Drop(0.25)',
            'Conv2D(64)×2 + BN + MaxPool + Drop(0.25)',
            'Conv2D(128)×2 + BN + GlobalAvgPool + Drop(0.4)',
            'Dense(256,ReLU) → Dense(128,ReLU)',
            'Softmax(4)'
        ]
    })

@app.route('/api/segments')
def segments():
    return jsonify({'segments': ROAD_SEGMENTS})

@app.route('/api/stats')
def stats():
    with _lock:
        n = len(_cache)
    return jsonify({
        'system': 'TrafficAI v2',
        'active_segments': n,
        'model': 'TensorFlow CNN' if USE_TF else 'NumPy Simulator',
        'chatbot': 'AIML 1.0 (40+ patterns)',
        'refresh_interval': 60,
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    print(f"\n{'='*50}")
    print(f"  TrafficAI v2 — Server Ready")
    print(f"  Dashboard : http://localhost:5000")
    print(f"  API       : http://localhost:5000/api/predict/all")
    print(f"  Model     : {'TensorFlow CNN' if USE_TF else 'NumPy Simulator'}")
    print(f"{'='*50}\n")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port, use_reloader=False)
