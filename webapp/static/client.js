const socket = io();
let stream = null;
let running = false;
const video = document.getElementById('video');
const img = document.getElementById('annotated');
const status = document.getElementById('status');
const fpsEl = document.getElementById('fps');
const kptsEl = document.getElementById('kpts');
const repCountEl = document.getElementById('repCount');
const repPhaseEl = document.getElementById('repPhase');
const repStatusEl = document.getElementById('repStatus');
const confidenceEl = document.getElementById('confidence');
const totalsPushupEl = document.getElementById('totalPushup');
const totalsSquatEl = document.getElementById('totalSquat');
const totalsJumpEl = document.getElementById('totalJump');
const recStart = document.getElementById('recStart');
const recStop = document.getElementById('recStop');
const sessionsList = document.getElementById('sessionsList');
const downloadSession = document.getElementById('downloadSession');
const athleteName = document.getElementById('athleteName');
const sportSelect = document.getElementById('sportSelect');
const modeSelect = document.getElementById('modeSelect');

socket.on('connect', () => { status.textContent = 'Connected'; });
socket.on('annotated', (data) => {
  console.log('Received annotated frame, kpts=', data.kpts);
  img.src = data.image;
  img.setAttribute('data-ts', Date.now());
  if (kptsEl) kptsEl.textContent = data.kpts || 0;
  if (confidenceEl && typeof data.confidence === 'number') {
    confidenceEl.textContent = data.confidence.toFixed(2);
    confidenceEl.parentElement.dataset.level = data.confidence >= 0.55 ? 'good' : (data.confidence >= 0.35 ? 'ok' : 'low');
  }
  if (data.counts) {
    const counts = data.counts;
    if (repCountEl && typeof counts.count === 'number') repCountEl.textContent = counts.count;
    if (repPhaseEl && counts.phase) repPhaseEl.textContent = counts.phase;
    if (repStatusEl && counts.status) repStatusEl.textContent = counts.status;
    if (counts.totals) {
      if (totalsPushupEl && typeof counts.totals.pushup === 'number') totalsPushupEl.textContent = counts.totals.pushup;
      if (totalsSquatEl && typeof counts.totals.squat === 'number') totalsSquatEl.textContent = counts.totals.squat;
      if (totalsJumpEl && typeof counts.totals.jump === 'number') totalsJumpEl.textContent = counts.totals.jump;
    }
  }
});
socket.on('status', (s) => {
  status.textContent = (s.state === 'ready') ? 'System Ready' : s.state;
  const startBtn = document.getElementById('start');
  // allow the user to start the local camera preview even while the
  // server is initializing the heavy detector. Only disable on error.
  if (s.state === 'ready') startBtn.disabled = false;
  if (s.state === 'error') startBtn.disabled = true;
});

// server shutdown acknowledgement
socket.on('shutdown_ack', (m) => {
  console.log('Server shutdown ack:', m);
  status.textContent = 'server shutting down';
  // optional visual feedback
  setTimeout(() => { status.textContent = 'disconnected'; }, 1000);
});

// Ctrl+Q to request server shutdown (with confirmation)
window.addEventListener('keydown', (ev) => {
  // ctrlKey may be true on Windows; also allow MetaKey for macOS
  if ((ev.ctrlKey || ev.metaKey) && ev.key && ev.key.toLowerCase() === 'q') {
    ev.preventDefault();
    const ok = confirm('Shutdown server? Press OK to request server shutdown (Ctrl+Q)');
    if (!ok) return;
    if (socket && socket.connected) {
      socket.emit('shutdown', { from: 'client' });
      status.textContent = 'shutdown requested';
    } else {
      alert('Not connected to server');
    }
  }
});

socket.on('record_status', (r) => {
  console.log('record status', r);
  if (r.state === 'recording') {
    recStart.disabled = true;
    recStop.disabled = false;
  } else {
    recStart.disabled = false;
    recStop.disabled = true;
    if (r.name) {
      // refresh session list
      fetch('/sessions').then(r => r.json()).then(data => {
        sessionsList.innerHTML = '';
        data.sessions.forEach(s => {
          const o = document.createElement('option'); o.value = s; o.textContent = s; sessionsList.appendChild(o);
        });
        if (sessionsList.options.length) {
          downloadSession.href = '/sessions/' + sessionsList.options[0].value;
        }
      });
    }
  }
});

async function start() {
  if (running) return;
  running = true;
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    video.srcObject = stream;
    // wait until metadata/dimensions are available before starting the
    // capture loop so canvas sizing and drawImage work reliably.
    await video.play();
    if (video.videoWidth === 0 || video.videoHeight === 0) {
      await new Promise((resolve) => {
        function _onMeta() {
          video.removeEventListener('loadedmetadata', _onMeta);
          resolve();
        }
        video.addEventListener('loadedmetadata', _onMeta);
        // fallback in case loadedmetadata doesn't fire quickly
        setTimeout(resolve, 1000);
      });
    }
    // start capturing frames once we have dimensions
    captureLoop();
  } catch (e) {
    // show error locally and send diagnostic info to server so logs capture it
    const msg = (e && e.message) ? e.message : String(e);
    console.error('Camera error:', e);
    status.textContent = 'camera error: ' + msg;
    // enumerate devices (may require permissions) to help debug
    try {
      const devices = await navigator.mediaDevices.enumerateDevices();
      console.log('Media devices:', devices);
      if (socket && socket.connected) {
        socket.emit('client_log', { type: 'camera_error', message: msg, devices: devices });
      }
    } catch (ed) {
      console.warn('enumerateDevices failed', ed);
      if (socket && socket.connected) socket.emit('client_log', { type: 'camera_error', message: msg, devices: null });
    }
    running = false;
  }
}

function stop() {
  running = false;
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  status.textContent = 'stopped';
}


async function captureLoop() {
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d');
  let last = performance.now();
  while (running) {
    try {
      // ensure the video has real dimensions before drawing/sending
      const vw = video.videoWidth || video.clientWidth;
      const vh = video.videoHeight || video.clientHeight;
      if (!vw || !vh) {
        // wait a short while and retry
        await new Promise(r => setTimeout(r, 100));
        continue;
      }
      // resize canvas to match actual video size for crisp frames
      if (canvas.width !== vw || canvas.height !== vh) {
        canvas.width = vw;
        canvas.height = vh;
      }
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
      const data = canvas.toDataURL('image/jpeg', 0.8);
      // send frame to server
      if (socket && socket.connected) {
        socket.emit('frame', {
          image: data,
          sport: sportSelect ? sportSelect.value : 'pushup',
        });
      }
      const now = performance.now();
      const dt = now - last;
      last = now;
      if (fpsEl) fpsEl.textContent = (1000/dt).toFixed(1);
    } catch (e) {
      console.error(e);
    }
    await new Promise(r => setTimeout(r, 100));
  }
}

document.getElementById('start').addEventListener('click', start);
document.getElementById('stop').addEventListener('click', stop);
if (recStart) recStart.addEventListener('click', () => {
  const sessionName = prompt('Session name (optional)');
  const meta = {
    name: athleteName ? athleteName.value : '',
    sport: sportSelect ? sportSelect.value : 'pushup',
    mode: modeSelect ? modeSelect.value : 'count'
  };
  socket.emit('record', { action: 'start', name: sessionName || undefined, meta: meta });
});

// demo animations rendered via dedicated HTML canvases served through iframes
const demoFrames = {
  pushup: { src: '/demos/push-up', title: 'Push-up demo' },
  squat: { src: '/demos/squat', title: 'Squat demo' },
  jump: { src: '/demos/jump', title: 'Jump demo' },
  jumping_jack: { src: '/demos/jumping-jack', title: 'Jumping Jack demo' },
  lunge: { src: '/demos/lunge', title: 'Lunge demo' }
};

function updateDemo(sport) {
  const container = document.getElementById('demoAnim');
  if (!container) return;
  const config = demoFrames[sport] || demoFrames.pushup;
  container.innerHTML = `<iframe src="${config.src}" title="${config.title}" class="demo-frame" loading="lazy" allowtransparency="true"></iframe>`;
}

if (sportSelect) {
  sportSelect.addEventListener('change', (e) => updateDemo(e.target.value));
  // initial
  updateDemo(sportSelect.value);
  sportSelect.addEventListener('change', () => {
    if (repCountEl) repCountEl.textContent = '0';
    if (repPhaseEl) repPhaseEl.textContent = 'idle';
    if (repStatusEl) repStatusEl.textContent = 'calibrating';
  });
}
if (recStop) recStop.addEventListener('click', () => socket.emit('record', { action: 'stop' }));
if (sessionsList) sessionsList.addEventListener('change', () => {
  const v = sessionsList.value;
  if (v) downloadSession.href = '/sessions/' + v;
});
