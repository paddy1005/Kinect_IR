// static/app.js
let pc, controlChannel;
const $ = (id)=>document.getElementById(id);

function logLine(...args){
  const pre = $("log");
  pre.textContent += args.join(' ') + '\n';
  pre.scrollTop = pre.scrollHeight;
}
function setBadge(id, on){
  $(id).textContent = (id==='poseState' ? 'pose: ' : 'rec: ') + (on ? 'on' : 'off');
  $(id).style.color = on ? '#60d394' : '';
}
function setStatus(state){
  const dot = $("statusDot");
  dot.classList.remove('dot--idle','dot--connecting','dot--ok','dot--err');
  dot.classList.add(
    state==='connecting' ? 'dot--connecting' :
    state==='ok' ? 'dot--ok' :
    state==='err' ? 'dot--err' : 'dot--idle'
  );
}
function sendCmd(cmd, extra={}){
  const token = $("token").value.trim();
  if(!controlChannel || controlChannel.readyState!=='open'){ logLine('datachannel not open'); return; }
  controlChannel.send(JSON.stringify({cmd, token, ...extra}));
  logLine('→', cmd, JSON.stringify(extra));
}

// === LUX UI ===
const luxBadge = document.getElementById("luxBadge");
const luxNowEl = document.getElementById("luxNow");
let _lastLuxTs = 0;

function updateLuxUI(v){
  if (!Number.isFinite(v)) return;
  if (luxBadge) luxBadge.textContent = `lux: ${v.toFixed(1)} lx`;
  if (luxNowEl) luxNowEl.textContent = `${v.toFixed(1)} lx`;
  _lastLuxTs = Date.now();
}

async function pollLuxHTTP(){
  try{
    const r = await fetch("/lux", { cache: "no-store" });
    const j = await r.json();
    if (Number.isFinite(j.lux)) updateLuxUI(j.lux);
  }catch(e){
    console.warn("pollLuxHTTP failed:", e);
  }
}

// 3秒更新が無ければHTTPでフォールバック
setInterval(()=>{ 
  if (Date.now() - _lastLuxTs > 3000) pollLuxHTTP(); 
}, 1000);

// === UI Wiring ===
$("mode").onchange = () => {
  const mode = $("mode").value;
  $("pctRow").style.display = (mode==="pct")?"flex":"none";
  $("absRow").style.display = (mode==="abs")?"flex":"none";
};
$("loPct").oninput = ()=>{ $("loPctVal").textContent = $("loPct").value; };
$("hiPct").oninput = ()=>{ $("hiPctVal").textContent = $("hiPct").value; };

$("applyPct").onclick = () => {
  const lo = parseFloat($("loPct").value);
  const hi = parseFloat($("hiPct").value);
  if(hi <= lo){ alert("hi は lo より大きくしてください"); return; }
  sendCmd("set_range_pct", { lo_pct: lo, hi_pct: hi });
};
$("applyAbs").onclick = () => {
  const lo = parseInt($("loAbs").value,10);
  const hi = parseInt($("hiAbs").value,10);
  if(!(lo>=0 && hi<=65535) || hi<=lo){ alert("0..65535 の範囲で lo < hi にしてください"); return; }
  sendCmd("set_range_abs", { lo_abs: lo, hi_abs: hi });
};
$("calib").onclick = () => sendCmd("calib_reload");
$("geom").onchange = () => sendCmd("set_geom", { enable: $("geom").checked });

$("poseStart").onclick = () => sendCmd("pose_start");
$("poseStop").onclick = () => sendCmd("pose_stop");
$("recStart").onclick = () => sendCmd("rec_start");
$("recStop").onclick = () => sendCmd("rec_stop");

$("roiApply").onclick = () => {
  const s = $("roi").value.trim();
  if(!s){ alert("x1,y1;...;x4,y4 を入力"); return; }
  try{
    const pts = s.split(";").map(p=>{
      const [x,y] = p.split(",").map(v=>parseFloat(v.trim()));
      if(!isFinite(x)||!isFinite(y)) throw new Error("NaN");
      return [x,y];
    });
    if(pts.length < 3){ alert("少なくとも3点必要です(推奨4点)"); return; }
    sendCmd("set_roi", { points: pts });
  }catch(e){
    alert("パースに失敗: " + e.message);
  }
};
$("roiClear").onclick = () => sendCmd("clear_roi");
$("clearLog").onclick = () => { $("log").textContent = ""; };

// === Connect ===
$("connect").onclick = async () => {
  try{
    setStatus('connecting');
    $("connect").disabled = true;
    $("token").disabled = true;

    pc = new RTCPeerConnection({ iceServers: [{ urls: 'stun:stun.l.google.com:19302' }] });
    pc.addTransceiver("video", { direction: "recvonly" });

    pc.ontrack = (ev) => {
      const video = $("v");
      if (ev.streams && ev.streams[0]) video.srcObject = ev.streams[0];
      else { const s = new MediaStream(); s.addTrack(ev.track); video.srcObject = s; }
    };

    controlChannel = pc.createDataChannel('control');
    controlChannel.onopen = ()=>{
      logLine("control channel: open");
      controlChannel.send(JSON.stringify({ cmd:"lux_poll", token:$("token").value.trim() }));
    };
    controlChannel.onclose = ()=> logLine("control channel: closed");

    controlChannel.onmessage = (e) => {
      const text = (typeof e.data === "string") ? e.data : "";
      console.log("RX:", text);
      try {
        const obj = JSON.parse(text);
        if (obj && typeof obj.lux === "number") {
          updateLuxUI(obj.lux);
          return;
        }
      } catch(_){}
      logLine("←", text);
    };

    const offer = await pc.createOffer();
    await pc.setLocalDescription(offer);

    const res = await fetch('/offer', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ sdp: offer.sdp, type: offer.type, token: $("token").value.trim() })
    });
    if(!res.ok){
      setStatus('err');
      logLine('offer failed', await res.text());
      $("connect").disabled = false;
      $("token").disabled = false;
      return;
    }
    const answer = await res.json();
    await pc.setRemoteDescription(answer);
    setStatus('ok');
    logLine('connected');
  }catch(err){
    setStatus('err');
    logLine('connect error:', String(err));
    $("connect").disabled = false;
    $("token").disabled = false;
  }
};
