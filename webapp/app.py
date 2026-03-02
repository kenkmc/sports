"""Flask + Socket.IO server that accepts base64-encoded frames from the browser,
runs the existing PoseDetector + SimpleTracker, and returns annotated frames.

Run:
    .\.venv\Scripts\Activate.ps1
    python webapp\app.py

Open in browser: http://localhost:5000
"""
import base64
import io
import os
import sys
from pathlib import Path
import argparse
import time
import threading
import json
import glob

# Ensure repo root is on sys.path so `from src.*` imports work when running this file
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from flask import Flask, render_template, abort
from flask import jsonify
from flask import send_from_directory
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import logging
try:
    from src.sheets_uploader import SheetsUploader
except Exception:
    SheetsUploader = None

from src.pose_utils import KeypointSmoother, average_visibility, keypoints_to_dict
from src.rep_counter import RepCounter

# Import PoseDetector and SimpleTracker lazily inside start_detector to avoid
# blocking server startup while MediaPipe loads large models.

app = Flask(__name__, template_folder='templates')
socketio = SocketIO(app, cors_allowed_origins='*')

pose_detector = None
tracker = None
# post-processing helpers
smoother = KeypointSmoother(alpha=0.25, visibility_alpha=0.2)
rep_counter = RepCounter()
# recording state
_recording = False
_current_session = None
_sessions_dir = os.path.join(ROOT, 'webapp', 'sessions')
os.makedirs(_sessions_dir, exist_ok=True)
LOG = logging.getLogger('webapp')
LOG.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
LOG.addHandler(handler)

# Language state
_language = 'en'

TRANSLATIONS = {
    'en': {
        'Sport': 'Sport',
        'Reps': 'Reps',
        'Status': 'Status',
        'Conf': 'Conf',
        'LOW_VISIBILITY': '! LOW VISIBILITY !',
        'CHECK_LIGHTING': 'Check lighting',
        'TOO_FAR': '! TOO FAR !',
        'TOO_CLOSE': '! TOO CLOSE !',
        'NO_PERSON': '! NO PERSON !'
    },
    'zh': {
        'Sport': '運動',
        'Reps': '次數',
        'Status': '狀態',
        'Conf': '信心',
        'LOW_VISIBILITY': '! 能見度低 !',
        'CHECK_LIGHTING': '檢查光線',
        'TOO_FAR': '! 太遠 !',
        'TOO_CLOSE': '! 太近 !',
        'NO_PERSON': '! 無人 !'
    }
}

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/health')
def health():
    """Simple health endpoint reporting whether the detector is ready."""
    ready = pose_detector is not None and tracker is not None
    return jsonify({'ready': ready, 'detector_loaded': bool(pose_detector is not None)})


@socketio.on('connect')
def on_connect():
    print('Client connected')
    emit('connected', {'data': 'ok'})
    # lazily initialize the heavy detector in a background thread so the server
    # can start quickly without blocking on MediaPipe model load.
    global pose_detector, tracker
    
    if pose_detector is not None:
        emit('status', {'state': 'ready'})

    if pose_detector is None:
        def _init():
            try:
                socketio.emit('status', {'state': 'initializing'})
                # Revert to Lite model (0) for better FPS.
                # We will rely on better user positioning (calibration) for accuracy.
                start_detector(complexity=0)
                socketio.emit('status', {'state': 'ready'})
                print('Pose detector initialized')
            except Exception as e:
                socketio.emit('status', {'state': 'error', 'message': str(e)})
                print('Error initializing pose detector:', e)

        t = threading.Thread(target=_init, daemon=True)
        t.start()


@socketio.on('frame')
def on_frame(data):
    # data: {'image': 'data:image/jpeg;base64,...', 'sport': 'pushup'}
    img_b64 = data.get('image', '').split(',', 1)[-1]
    sport = data.get('sport', 'pushup') or 'pushup'
    try:
        img_bytes = base64.b64decode(img_b64)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError('bad frame')
        
        # Resize to speed up processing
        h, w = frame.shape[:2]
        if w > 640:
            scale = 640 / w
            new_h = int(h * scale)
            frame = cv2.resize(frame, (640, new_h))
    except Exception as e:
        emit('error', {'message': 'invalid image', 'exc': str(e)})
        return

    # ensure detector initialized
    if pose_detector is None or tracker is None:
        emit('error', {'message': 'detector not ready'})
        return

    # process
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    keypoints, landmarks_obj = pose_detector.detect(rgb)
    kp_dict = {}
    smoothed = {}
    confidence = 0.0
    rep_info = rep_counter.snapshot(sport)
    bb = None
    h, w = frame.shape[:2]
    if keypoints:
        kp_dict = keypoints_to_dict(keypoints)
        smoothed = smoother.smooth(kp_dict)
        required = rep_counter.required_landmarks(sport)
        confidence = average_visibility(smoothed, required=required) if required else average_visibility(smoothed)
        rep_info = rep_counter.update(sport, smoothed, confidence=confidence)
        # compute bbox similar to demo
        xs = [l[0] for l in keypoints]
        ys = [l[1] for l in keypoints]
        x_min = max(0.0, min(xs) - 0.05)
        y_min = max(0.0, min(ys) - 0.05)
        x_max = min(1.0, max(xs) + 0.05)
        y_max = min(1.0, max(ys) + 0.05)
        x = int(x_min * w)
        y = int(y_min * h)
        ww = int((x_max - x_min) * w)
        hh = int((y_max - y_min) * h)
        bb = (x, y, ww, hh)
        detections = [bb]
    else:
        detections = []

    tracks = tracker.update(detections)

    # LOG.info('Received frame for processing [sport=%s conf=%.2f count=%s]', sport, confidence, rep_info.get('count'))
    # draw annotations
    if bb:
        x, y, ww, hh = bb
        cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
    for tid, tbb in tracks:
        x, y, ww, hh = tbb
        cv2.rectangle(frame, (x, y), (x + ww, y + hh), (255, 128, 0), 2)
        cv2.putText(frame, str(tid), (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    if landmarks_obj is not None:
        for lm in landmarks_obj.landmark:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

    # overlay metrics text
    t = TRANSLATIONS.get(_language, TRANSLATIONS['en'])
    overlay_lines = [
        f"{t['Sport']}: {sport}",
        f"{t['Reps']}: {rep_info.get('count', 0)} ({rep_info.get('phase', '-')})",
        f"{t['Status']}: {rep_info.get('status', '')}",
        f"{t['Conf']}: {confidence:.2f}"
    ]
    
    # Calibration / Visibility Feedback
    if confidence < 0.5:
        overlay_lines.append(t['LOW_VISIBILITY'])
        overlay_lines.append(t['CHECK_LIGHTING'])
    
    # Check if user is too close or too far (based on bounding box size relative to frame)
    if bb:
        _, _, ww, hh = bb
        # Frame height is h
        if hh < h * 0.3:
            overlay_lines.append(t['TOO_FAR'])
        elif hh > h * 0.9:
            overlay_lines.append(t['TOO_CLOSE'])
    else:
        overlay_lines.append(t['NO_PERSON'])

    base_y = 30
    for line in overlay_lines:
        cv2.putText(frame, line, (20, base_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2)
        base_y += 24

    # encode as JPEG and send back
    _, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
    # save last annotated for debugging
    # try:
    #     with open(os.path.join(ROOT, 'webapp', 'last_annotated.jpg'), 'wb') as f:
    #         f.write(jpeg.tobytes())
    #     LOG.info('Saved last_annotated.jpg')
    # except Exception as e:
    #     LOG.exception('Failed to save last annotated')
    b64 = base64.b64encode(jpeg.tobytes()).decode('ascii')
    emit('annotated', {
        'image': 'data:image/jpeg;base64,' + b64,
        'kpts': len(keypoints),
        'counts': rep_info,
        'confidence': confidence,
        'sport': sport,
    })
    # LOG.info('Emitted annotated frame back to client; kpts=%d reps=%s', len(keypoints), rep_info.get('count'))

    # if recording, append a compact JSON line with timestamp and keypoints
    try:
        if _recording and _current_session:
            import json
            rec = {
                'ts': time.time(),
                'kpts': keypoints,
                'counts': rep_info,
                'confidence': confidence,
                'sport': sport,
            }
            with open(_current_session, 'a', encoding='utf8') as fh:
                fh.write(json.dumps(rec) + '\n')
    except Exception:
        LOG.exception('Failed to write recording frame')


@socketio.on('client_log')
def on_client_log(data):
    # simple relay of client-side diagnostic messages to the server log
    try:
        LOG.info('Client log: %s', str(data))
    except Exception:
        LOG.exception('Failed to log client message')


@socketio.on('set_language')
def on_set_language(data):
    global _language
    lang = data.get('lang', 'en')
    if lang in TRANSLATIONS:
        _language = lang
        LOG.info('Language set to %s', _language)


@socketio.on('shutdown')
def on_shutdown(msg=None):
    """Requested by client: attempt a graceful shutdown of the server.
    Emits an ack back to the requesting client, then shuts down the process
    in a background thread after a short delay to allow the ack to be sent.
    """
    try:
        LOG.info('Shutdown requested by client: %s', msg)
        emit('shutdown_ack', {'status': 'shutting_down'})

        def _quit_after_delay():
            # give the ack a moment to be delivered
            time.sleep(0.5)
            try:
                LOG.info('Exiting process due to client shutdown request')
                # Try SocketIO stop first (best-effort)
                try:
                    socketio.stop()
                except Exception:
                    pass
                # Fallback to os._exit to ensure process termination
                os._exit(0)
            except Exception:
                LOG.exception('Error during shutdown')

        t = threading.Thread(target=_quit_after_delay, daemon=True)
        t.start()
    except Exception:
        LOG.exception('Failed to handle shutdown request')


@socketio.on('record')
def on_record(msg):
    """Client requests recording start/stop. msg: {'action': 'start'|'stop', 'name': optional}"""
    global _recording, _current_session
    action = msg.get('action') if isinstance(msg, dict) else None
    if action == 'start':
        name = msg.get('name') or f'session_{int(time.time())}.jsonl'
        path = os.path.join(_sessions_dir, name)
        try:
            # create/overwrite and write metadata header as first JSON line
            meta = msg.get('meta') if isinstance(msg, dict) else {}
            with open(path, 'w', encoding='utf8') as fh:
                import json
                fh.write(json.dumps({'meta': meta, 'started': time.time()}) + '\n')
            _current_session = path
            _recording = True
            emit('record_status', {'state': 'recording', 'name': name})
            LOG.info('Started recording to %s meta=%s', path, meta)
        except Exception as e:
            emit('record_status', {'state': 'error', 'message': str(e)})
    elif action == 'stop':
        # finalize
        _recording = False
        sess_path = _current_session
        name = os.path.basename(sess_path) if sess_path else None
        _current_session = None
        emit('record_status', {'state': 'stopped', 'name': name})
        LOG.info('Stopped recording, saved %s', name)
        # try to upload a summary row to Google Sheets if configured
        try:
            if SheetsUploader is not None:
                sheet_id = os.environ.get('SHEETS_ID')
                creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
                if sheet_id and creds:
                    uploader = SheetsUploader(sheet_id=sheet_id, creds_path=creds)
                    # read metadata header for summary
                    import json
                    with open(sess_path, 'r', encoding='utf8') as fh:
                        first = fh.readline()
                        hdr = json.loads(first)
                    summary = {
                        'name': hdr.get('meta', {}).get('name', ''),
                        'sport': hdr.get('meta', {}).get('sport', ''),
                        'mode': hdr.get('meta', {}).get('mode', ''),
                        'session_file': name,
                        'timestamp': hdr.get('started', '')
                    }
                    ok = uploader.upload_record(summary)
                    LOG.info('Uploaded session summary to sheets: %s', ok)
        except Exception:
            LOG.exception('Failed to upload session summary')
    else:
        emit('record_status', {'state': 'error', 'message': 'unknown action'})


@app.route('/sessions')
def list_sessions():
    files = sorted(os.listdir(_sessions_dir))
    return jsonify({'sessions': files})


@app.route('/sessions/<name>')
def get_session(name):
    # serve session file for download
    return send_from_directory(_sessions_dir, name, as_attachment=True)


@app.route('/sessions/<name>/json')
def get_session_json(name):
    path = os.path.join(_sessions_dir, name)
    if not os.path.exists(path):
        return jsonify({'error': 'not found'}), 404
    import json
    recs = []
    try:
        with open(path, 'r', encoding='utf8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    # skip malformed lines
                    continue
    except Exception:
        return jsonify({'error': 'read failed'}), 500
    return jsonify({'records': recs})


@app.route('/viewer')
def viewer():
    return render_template('viewer.html')


DEMO_FILES = {
    'push-up': 'push-up.html',
    'pushup': 'push-up.html',
    'squat': 'squat.html',
    'jump': 'jump.html',
    'jumping-jack': 'jumping-jack.html',
    'jumping_jack': 'jumping-jack.html',
    'lunge': 'lunge.html',
}


@app.route('/demos/<name>')
def serve_demo(name):
    """Serve interactive canvas demos for the sport selector."""
    filename = DEMO_FILES.get(name)
    if not filename:
        abort(404)
    return send_from_directory(ROOT, filename)


@app.route('/leaderboard')
def leaderboard():
    data = get_leaderboard_data()
    return render_template('leaderboard.html', leaderboard=data)

def get_leaderboard_data():
    leaderboard = {'pushup': {}, 'squat': {}, 'jump': {}, 'jumping_jack': {}, 'lunge': {}}
    files = glob.glob(os.path.join(_sessions_dir, '*.jsonl'))
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf8') as f:
                # Read first line for metadata
                first_line = f.readline()
                if not first_line: continue
                try:
                    meta_data = json.loads(first_line)
                    meta = meta_data.get('meta', {})
                    name = meta.get('name', 'Anonymous')
                except json.JSONDecodeError:
                    continue
                
                if not name: name = 'Anonymous'
                
                # Read last line for final counts
                last_line = None
                for line in f:
                    if line.strip():
                        last_line = line
                
                if not last_line: continue
                
                try:
                    record = json.loads(last_line)
                    counts = record.get('counts', {})
                    totals = counts.get('totals', {})
                    
                    for sport in ['pushup', 'squat', 'jump', 'jumping_jack', 'lunge']:
                        score = totals.get(sport, 0)
                        if score > 0:
                            current_max = leaderboard[sport].get(name, 0)
                            if score > current_max:
                                leaderboard[sport][name] = score
                except json.JSONDecodeError:
                    continue
                        
        except Exception as e:
            LOG.error(f"Error parsing session {fpath}: {e}")
            continue
            
    # Convert to list format
    result = {}
    for sport, players in leaderboard.items():
        sorted_players = [{'name': k, 'score': v} for k, v in players.items()]
        sorted_players.sort(key=lambda x: x['score'], reverse=True)
        result[sport] = sorted_players
        
    return result

def start_detector(complexity=0):
    global pose_detector, tracker
    # import here to avoid heavy imports at module import time
    from src.pose_detector import PoseDetector
    from src.tracker import SimpleTracker
    pose_detector = PoseDetector(model_complexity=complexity, smooth=True)
    tracker = SimpleTracker(iou_threshold=0.3)
    smoother.reset()
    rep_counter.reset()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5000)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print('Starting webapp on http://%s:%s' % (args.host, args.port))
    # Use eventlet for SocketIO server
    socketio.run(app, host=args.host, port=args.port)

