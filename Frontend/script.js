const scene = document.getElementById('scene');
const door = document.getElementById('door');
const doorFrame = document.getElementById('doorFrame');
const requestBtn = document.getElementById('requestBtn');
const toggleDoorBtn = document.getElementById('toggleDoorBtn');
const toggleLightBtn = document.getElementById('toggleLightBtn');
const accessStatus = document.getElementById('accessStatus');
const helperText = document.getElementById('helperText');
const doorStatDot = document.getElementById('doorStatDot');
const doorStatText = document.getElementById('doorStatText');

let isDoorOpen = false;
let isLightOn = false;

const SCENE_STATES = ['requesting', 'access-granted', 'access-denied'];

// ── State setters ──────────────────────────────────────
function setDoorState(open) {
  isDoorOpen = open;
  door.classList.toggle('open', open);
  doorFrame.classList.toggle('door-open-state', open);
  refreshStat();
}

function setLightState(on) {
  isLightOn = on;
  scene.classList.toggle('lights-on', on);
  refreshStat();
}

function setSceneState(state) {
  SCENE_STATES.forEach(cls => scene.classList.remove(cls));
  if (state) scene.classList.add(state);
}

function setStatus(stateClass, text) {
  accessStatus.className = `status-pill ${stateClass}`;
  accessStatus.textContent = text;
}

function setHelper(text, cls = '') {
  helperText.textContent = text;
  helperText.className = `helper ${cls}`;
}

function refreshStat() {
  const dLabel = isDoorOpen ? 'Open' : 'Closed';
  const lLabel = isLightOn ? 'On' : 'Off';
  doorStatText.textContent = `Door: ${dLabel}  ·  Light: ${lLabel}`;
  doorStatDot.className = 'door-stat-dot' + (isDoorOpen ? ' open' : '');
}

// ── Backend request ────────────────────────────────────
async function requestVerification() {
  setSceneState('requesting');
  setStatus('pending', 'Requesting…');
  setHelper('Contacting the backend for verification…', 'pending');
  requestBtn.disabled = true;

  try {
    const res = await fetch('http://localhost:5000/verify', { method: 'POST' });
    if (!res.ok) throw new Error('non-ok');
    const data = await res.json();

    if (data.verified) {
      setSceneState('access-granted');
      setStatus('granted', 'Access Granted');
      setHelper('✅ Verified. Door unlocked and lights on.', 'granted');
      setDoorState(true);
      setLightState(true);
    } else {
      setSceneState('access-denied');
      setStatus('denied', 'Access Denied');
      setHelper('⛔ Verification failed. Access denied.', 'denied');
      setDoorState(false);
      setLightState(false);
    }
  } catch {
    setSceneState('access-denied');
    setStatus('denied', 'Backend Offline');
    setHelper('⚠️ Backend unreachable. Start the API and retry.', 'denied');
    setDoorState(false);
    setLightState(false);
  } finally {
    requestBtn.disabled = false;
    // Clear glow after delay
    setTimeout(() => {
      if (!scene.classList.contains('requesting')) {
        setSceneState(null);
      }
    }, 3000);
  }
}

// ── Event listeners ────────────────────────────────────
requestBtn.addEventListener('click', requestVerification);
toggleDoorBtn.addEventListener('click', () => setDoorState(!isDoorOpen));
toggleLightBtn && toggleLightBtn.addEventListener('click', () => setLightState(!isLightOn));
door.addEventListener('click', () => setDoorState(!isDoorOpen));

// Init
refreshStat();
