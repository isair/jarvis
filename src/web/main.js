const hostEl = document.getElementById('host');
const portEl = document.getElementById('port');
const inputEl = document.getElementById('input');
const outEl = document.getElementById('out');
const saveBtn = document.getElementById('save');
const sendBtn = document.getElementById('send');
const vstartBtn = document.getElementById('vstart');
const vstopBtn = document.getElementById('vstop');
const autospeakEl = document.getElementById('autospeak');
const warnEl = document.getElementById('warn');

function getBase(){
  const h = localStorage.getItem('jarvis_host') || '';
  const p = localStorage.getItem('jarvis_port') || '';
  return h && p ? `http://${h}:${p}` : '';
}

function load(){
  hostEl.value = localStorage.getItem('jarvis_host') || '';
  portEl.value = localStorage.getItem('jarvis_port') || '8756';
}

saveBtn.onclick = () => {
  localStorage.setItem('jarvis_host', hostEl.value.trim());
  localStorage.setItem('jarvis_port', portEl.value.trim());
  alert('Saved');
};

function speak(text){
  try{
    if(!('speechSynthesis' in window)) return;
    const u = new SpeechSynthesisUtterance(text);
    u.lang = navigator.language || 'en-US';
    window.speechSynthesis.cancel();
    window.speechSynthesis.speak(u);
  }catch(e){}
}

async function send(){
  const base = getBase();
  if(!base){ alert('Set host and port first.'); return; }
  const text = inputEl.value.trim();
  if(!text){ return; }
  sendBtn.disabled = true;
  outEl.textContent = '...';
  try{
    const res = await fetch(`${base}/api/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await res.json();
    const reply = data.text || '';
    outEl.textContent = reply;
    if(autospeakEl.checked && reply) speak(reply);
  }catch(e){
    outEl.textContent = 'Network error';
  } finally {
    sendBtn.disabled = false;
  }
}

sendBtn.onclick = send;
load();

// Voice input via Web Speech API (requires secure context on most browsers)
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
if(!SR){
  warnEl.textContent = 'Voice input not supported on this browser. Use text or HTTPS.';
  vstartBtn.disabled = true;
} else {
  rec = new SR();
  rec.continuous = false;
  rec.interimResults = false;
  rec.lang = navigator.language || 'en-US';
  rec.onresult = (e) => {
    try{
      const t = Array.from(e.results).map(r=>r[0]&&r[0].transcript||'').join(' ').trim();
      inputEl.value = t;
      send();
    }catch(err){}
  };
  rec.onend = () => {
    vstartBtn.disabled = false;
    vstopBtn.disabled = true;
  };
  rec.onerror = () => {
    vstartBtn.disabled = false;
    vstopBtn.disabled = true;
  };
  vstartBtn.onclick = () => {
    try{ rec.start(); vstartBtn.disabled = true; vstopBtn.disabled = false; }catch(e){}
  };
  vstopBtn.onclick = () => {
    try{ rec.stop(); }catch(e){}
  };
  if(location.protocol !== 'https:'){
    warnEl.textContent = 'Voice input may require HTTPS depending on your browser.';
  }
}

if('serviceWorker' in navigator){
  navigator.serviceWorker.register('/sw.js').catch(()=>{});
}


