async function fetchSessions() {
  const res = await fetch('/sessions');
  const j = await res.json();
  const sel = document.getElementById('sessions');
  sel.innerHTML = '';
  j.sessions.forEach(s => {
    const o = document.createElement('option'); o.value = s; o.textContent = s; sel.appendChild(o);
  });
}

async function loadSession(name) {
  const res = await fetch(`/sessions/${name}/json`);
  const j = await res.json();
  if (j.error) { alert(j.error); return; }
  const recs = j.records;
  // build arrays
  const ts = [];
  const kpts = [];
  const reps = [];
  recs.forEach(r => {
    if (r.meta) {
      // skip header
      return;
    }
    ts.push(new Date((r.ts || Date.now()) * 1000).toISOString());
    kpts.push((r.kpts && r.kpts.length) ? r.kpts.length : 0);
    reps.push(r.counts && typeof r.counts.count === 'number' ? r.counts.count : null);
  });
  drawChart(ts, kpts, reps, name);
  updateSummary(recs, name);
}

let chart = null;
function drawChart(labels, keypointData, repData, title) {
  const ctx = document.getElementById('chart').getContext('2d');
  if (chart) chart.destroy();
  chart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: labels,
      datasets: [
        { label: 'keypoints per frame', data: keypointData, borderColor: 'orange', backgroundColor: 'rgba(255,165,0,0.2)', tension: 0.2, yAxisID: 'y' },
        { label: 'rep count', data: repData, borderColor: '#4ade80', backgroundColor: 'rgba(74,222,128,0.2)', spanGaps: true, stepped: true, yAxisID: 'y1' }
      ]},
    options: {
      responsive: true,
      plugins: { title: { display: true, text: title } },
      scales: {
        x: { display: true },
        y: { display: true, title: { display: true, text: 'keypoints' } },
        y1: { display: true, position: 'right', title: { display: true, text: 'reps' }, grid: { drawOnChartArea: false } }
      }
    }
  });
}

function updateSummary(records, name) {
  const summary = document.getElementById('summary');
  if (!summary) return;
  const header = records.find(r => r.meta);
  const last = [...records].reverse().find(r => r.counts);
  const totals = last && last.counts && last.counts.totals ? last.counts.totals : {};
  summary.innerHTML = `
    <h3>${name}</h3>
    <p><strong>Athlete:</strong> ${(header && header.meta && header.meta.name) || 'n/a'}</p>
    <p><strong>Sport:</strong> ${(header && header.meta && header.meta.sport) || 'n/a'} | <strong>Mode:</strong> ${(header && header.meta && header.meta.mode) || 'n/a'}</p>
    <p><strong>Total reps</strong> — Push-up: ${totals.pushup || 0}, Squat: ${totals.squat || 0}, Jump: ${totals.jump || 0}</p>
  `;
}

document.getElementById('load').addEventListener('click', () => {
  const sel = document.getElementById('sessions');
  if (sel.value) loadSession(sel.value);
});

fetchSessions();
