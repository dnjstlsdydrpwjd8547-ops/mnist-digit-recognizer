const pad = document.getElementById("pad");
const ctx = pad.getContext("2d");
const statusEl = document.getElementById("status");
const goBtn = document.getElementById("go");
let drawing = false, dirty = false;

function reset() {
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, pad.width, pad.height);
  dirty = false;
  setStatus("숫자를 그려주세요.");
  document.getElementById("result").hidden = true;
}
function setStatus(msg, err = false) {
  statusEl.textContent = msg;
  statusEl.classList.toggle("err", err);
}
function pos(e) {
  const r = pad.getBoundingClientRect();
  return [(e.clientX - r.left) * pad.width / r.width, (e.clientY - r.top) * pad.height / r.height];
}
pad.addEventListener("pointerdown", e => {
  drawing = true; dirty = true; pad.setPointerCapture(e.pointerId);
  const [x, y] = pos(e); ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + 0.1, y); ctx.stroke();
});
pad.addEventListener("pointermove", e => {
  if (!drawing) return;
  const [x, y] = pos(e); ctx.lineTo(x, y); ctx.stroke();
});
["pointerup", "pointercancel", "pointerleave"].forEach(t => pad.addEventListener(t, () => drawing = false));

ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#000";
reset();
ctx.lineWidth = 20; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#000";

document.getElementById("clear").addEventListener("click", reset);
goBtn.addEventListener("click", async () => {
  if (!dirty) { setStatus("먼저 숫자를 그려주세요.", true); return; }
  goBtn.disabled = true; setStatus("인식 중...");
  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: pad.toDataURL("image/png") }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "서버 오류");
    show(data);
    setStatus("인식 완료");
  } catch (err) {
    setStatus("오류: " + err.message, true);
  } finally {
    goBtn.disabled = false;
  }
});
function show(d) {
  document.getElementById("digit").textContent = d.digit;
  document.getElementById("conf").textContent = `확률 ${(d.confidence * 100).toFixed(1)}%`;
  const ul = document.getElementById("bars");
  ul.innerHTML = "";
  d.probabilities.forEach((p, i) => {
    const li = document.createElement("li");
    if (i === d.digit) li.className = "top";
    li.innerHTML = `<span>${i}</span><div class="bar"><i style="width:${(p * 100).toFixed(1)}%"></i></div><span>${(p * 100).toFixed(1)}%</span>`;
    ul.appendChild(li);
  });
  document.getElementById("result").hidden = false;
}
