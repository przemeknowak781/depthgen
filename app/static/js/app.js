import { Viewer } from './viewer.js';

const $ = id => document.getElementById(id);
const viewer = new Viewer($('view'));
window.viewer = viewer;   // dostęp z konsoli do diagnostyki

// Parametry sterujące — nazwa pola == nazwa parametru w backendzie.
const PREP = ['deblock', 'chroma', 'sr_model', 'work_max'];
const RANGES = ['deblock', 'chroma', 'work_max',
  'input_size', 'tile_blend', 'clip_low', 'clip_high', 'gamma', 'contrast',
  'highlights', 'shadows',
  'bilateral', 'smooth', 'detail', 'detail_radius', 'detail_guard', 'detail_clamp',
  'micro', 'micro_radius', 'floor',
  'floor_soft', 'edge_falloff', 'corner', 'margin', 'alpha_threshold', 'alpha_grow',
  'cut_level', 'min_island', 'resolution', 'width_mm', 'relief_mm', 'base_mm', 'exres'];
const CHECKS = ['invert', 'trim', 'solid', 'alpha_cut'];
const SELECTS = ['shape', 'median', 'sr_model'];
// zmiany tych pól wymagają ponownego liczenia sieci, nie tylko przebudowy siatki
const DEPTH_ONLY = new Set(['input_size', 'tiles', 'tile_blend', 'model', ...PREP]);
// te w ogóle nie dotyczą podglądu (używane dopiero przy eksporcie)
const NO_REBUILD = new Set(['exres']);

const DEC = { gamma: 2, contrast: 2, tile_blend: 2, micro: 2, floor: 2, floor_soft: 3,
  edge_falloff: 2, corner: 2, margin: 2, detail: 2, smooth: 1, detail_radius: 1,
  micro_radius: 1, relief_mm: 1, base_mm: 1, clip_low: 1, clip_high: 1,
  detail_guard: 2, detail_clamp: 3, alpha_threshold: 2, cut_level: 2, min_island: 2,
  deblock: 2, chroma: 2, highlights: 2, shadows: 2 };

const state = { id: null, hasDepth: false, busy: false, pending: false, ctrl: null };

function params() {
  const p = {};
  for (const k of RANGES) p[k] = parseFloat($(k).value);
  for (const k of CHECKS) p[k] = $(k).checked;
  for (const k of SELECTS) p[k] = $(k).value;
  return p;
}

function label(k) {
  const b = $('v_' + k);
  if (!b) return;
  const v = parseFloat($(k).value);
  b.textContent = k in DEC ? v.toFixed(DEC[k]) : String(v);
}

/** Parametry sieci nie przeliczają się same — trzeba kliknąć „Generuj". */
function needsDepth(on) {
  if (!state.hasDepth && on) return;
  $('gen').classList.toggle('needs', on);
  $('gen').textContent = on ? 'Generuj mapę głębi ● zmiany czekają' : 'Generuj mapę głębi';
}

function busy(on, txt = '') {
  $('busy').classList.toggle('hidden', !on);
  $('busyTxt').textContent = txt;
}

// ---------- podgląd siatki ----------
let timer = null;
function schedule(delay = 200) {
  if (!state.hasDepth) return;
  if (!$('live').checked) return;
  clearTimeout(timer);
  timer = setTimeout(rebuild, delay);
}

async function rebuild() {
  if (!state.hasDepth) return;
  if (state.busy) { state.pending = true; return; }
  state.busy = true;
  state.ctrl?.abort();
  const ctrl = new AbortController();
  state.ctrl = ctrl;
  busy(true, 'Buduję siatkę…');
  const p = params();
  try {
    const r = await fetch('/api/mesh', {
      method: 'POST', signal: ctrl.signal,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: state.id, params: p, resolution: p.resolution }),
    });
    if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
    const buf = await r.arrayBuffer();
    viewer.load(buf);
    $('stats').textContent =
      `${(+r.headers.get('X-Mesh-Vertices')).toLocaleString('pl')} wierzchołków · ` +
      `${(+r.headers.get('X-Mesh-Faces')).toLocaleString('pl')} trójkątów · ` +
      `${r.headers.get('X-Mesh-Size').replaceAll(',', ' × ')} mm`;
    refreshHeightPreview(p);
  } catch (e) {
    if (e.name !== 'AbortError') $('prog').textContent = 'Błąd podglądu: ' + e.message;
  } finally {
    busy(false);
    state.busy = false;
    if (state.pending) { state.pending = false; rebuild(); }
  }
}

let hmTimer = null;
function refreshHeightPreview(p) {
  clearTimeout(hmTimer);
  hmTimer = setTimeout(async () => {
    const r = await fetch('/api/heightmap', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: state.id, params: p }),
    });
    if (!r.ok) return;
    const url = URL.createObjectURL(await r.blob());
    const img = $('mapDepth');
    if (img.dataset.url) URL.revokeObjectURL(img.dataset.url);
    img.dataset.url = url;
    img.src = url;
  }, 350);
}

// ---------- wgrywanie obrazu ----------
async function upload(file) {
  const fd = new FormData();
  fd.append('file', file);
  busy(true, 'Wczytuję obraz…');
  try {
    const r = await fetch('/api/upload', { method: 'POST', body: fd });
    if (!r.ok) throw new Error(await r.text());
    const j = await r.json();
    state.id = j.id;
    state.hasDepth = false;
    const url = URL.createObjectURL(file);
    $('thumb').src = url;
    $('mapSrc').src = url;
    $('drop').classList.add('has');
    $('gen').disabled = false;
    $('export').disabled = true;
    $('saveDepth').disabled = true;
    $('prog').textContent = `${j.width} × ${j.height} px — gotowe do analizy głębi.`;
    $('hint').textContent = 'Kliknij „Generuj mapę głębi”.';
    if (j.blockiness > 1.12) {
      if (parseFloat($('deblock').value) === 0) {
        // powyżej ~0,7 filtr zaczyna zjadać prawdziwą fakturę, więc tam się zatrzymujemy
        $('deblock').value = Math.min(0.7, 0.25 + (j.blockiness - 1.12) * 0.35).toFixed(2);
        label('deblock');
        save();
      }
      $('prepInfo').textContent =
        `Wykryto artefakty JPEG (blokowość ${j.blockiness.toFixed(2)}) — ` +
        `włączyłem czyszczenie. Przy mocno zniszczonym obrazie dodaj upscaling 4×.`;
      $('deblock').closest('.grp').classList.add('open');
    } else {
      $('prepInfo').textContent = `Obraz czysty (blokowość ${j.blockiness.toFixed(2)}).`;
    }
    if (j.has_alpha) {
      $('alpha_cut').checked = true;
      $('trim').checked = true;
      $('alphaInfo').textContent = 'Wykryto przezroczyste tło — wycinanie sylwetki włączone.';
      $('alpha_cut').closest('.grp').classList.add('open');
      save();
    } else {
      $('alphaInfo').textContent = 'Brak przezroczystości — tnij progiem wysokości poniżej.';
    }
  } catch (e) {
    $('prog').textContent = 'Błąd wgrywania: ' + e.message;
  } finally { busy(false); }
}

// ---------- generowanie głębi ----------
async function generate() {
  if (!state.id) return;
  $('gen').disabled = true;
  busy(true, 'Analiza głębi…');
  const poll = setInterval(async () => {
    try {
      const r = await fetch('/api/progress/' + state.id);
      const j = await r.json();
      if (j.msg) { $('prog').textContent = j.msg; $('busyTxt').textContent = j.msg; }
    } catch { /* ignoruj */ }
  }, 500);
  const t0 = performance.now();
  try {
    const r = await fetch('/api/depth', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        id: state.id,
        model: $('model').value,
        input_size: parseInt($('input_size').value),
        tiles: parseInt($('tiles').value),
        tile_blend: parseFloat($('tile_blend').value),
        prep: {
          deblock: parseFloat($('deblock').value),
          chroma: parseFloat($('chroma').value),
          sr_model: $('sr_model').value,
          work_max: parseInt($('work_max').value),
        },
      }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const j = await r.json();
    state.hasDepth = true;
    needsDepth(false);
    $('export').disabled = false;
    $('saveDepth').disabled = false;
    $('hint').classList.add('hidden');
    $('prog').textContent = `Mapa głębi gotowa w ${(j.ms / 1000).toFixed(1)} s ` +
      `(${(performance.now() - t0) / 1000 | 0} s łącznie).`;
    if (j.prep) {
      const p = j.prep;
      const d = (p.blockiness_before - p.blockiness_after) / Math.max(p.blockiness_before - 1, 1e-6);
      $('prepInfo').textContent =
        `Obraz roboczy ${p.width}×${p.height} px` +
        (p.sr_input ? ` (upscaling ${p.sr_input[0]}×${p.sr_input[1]} → ${p.scale}×)` : '') +
        ` · blokowość ${p.blockiness_before.toFixed(2)} → ${p.blockiness_after.toFixed(2)}` +
        (p.blockiness_before > 1.05 ? ` (−${Math.round(Math.max(0, Math.min(1, d)) * 100)}%)` : '');
      // podgląd obrazu po czyszczeniu
      $('mapSrc').src = `/api/image/${state.id}?v=${Date.now()}`;
    }
    await rebuild();
    viewer.fit();
  } catch (e) {
    $('prog').textContent = 'Błąd: ' + e.message;
  } finally {
    clearInterval(poll);
    $('gen').disabled = false;
    busy(false);
  }
}

// ---------- eksport ----------
async function doExport() {
  const p = params();
  $('export').disabled = true;
  busy(true, 'Buduję siatkę eksportową…');
  $('exInfo').textContent = 'To może chwilę potrwać przy wysokiej rozdzielczości…';
  try {
    const r = await fetch('/api/export', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ id: state.id, params: p, resolution: p.exres, format: $('format').value }),
    });
    if (!r.ok) throw new Error((await r.json()).detail || r.statusText);
    const blob = await r.blob();
    const cd = r.headers.get('Content-Disposition') || '';
    // filename* niesie pełną nazwę w UTF-8; filename="" to zapasowa wersja ASCII
    const utf8 = cd.match(/filename\*=UTF-8''([^;]+)/);
    const name = utf8 ? decodeURIComponent(utf8[1])
      : (cd.match(/filename="(.+?)"/) || [, 'relief.' + $('format').value])[1];
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
    $('exInfo').textContent =
      `Zapisano ${name} — ${(+r.headers.get('X-Mesh-Faces')).toLocaleString('pl')} trójkątów, ` +
      `${(blob.size / 1048576).toFixed(1)} MB.`;
  } catch (e) {
    $('exInfo').textContent = 'Błąd eksportu: ' + e.message;
  } finally {
    $('export').disabled = false;
    busy(false);
  }
}

async function saveHeight() {
  const r = await fetch('/api/heightmap', {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ id: state.id, params: params() }),
  });
  if (!r.ok) return;
  const a = document.createElement('a');
  a.href = URL.createObjectURL(await r.blob());
  a.download = 'heightmap_16bit.png';
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}

// ---------- podpięcie UI ----------
for (const k of [...RANGES, ...CHECKS, ...SELECTS, 'tiles', 'model']) {
  const el = $(k);
  if (!el) continue;
  label(k);
  const ev = el.type === 'range' ? 'input' : 'change';
  el.addEventListener(ev, () => {
    label(k);
    save();
    if (NO_REBUILD.has(k)) return;
    if (DEPTH_ONLY.has(k)) needsDepth(true);
    else schedule(el.type === 'range' ? 220 : 60);
  });
}

// ---------- presety i zapamiętywanie ustawień ----------
const ALL = [...RANGES, ...CHECKS, ...SELECTS, 'tiles'];
const DEFAULTS = Object.fromEntries(ALL.map(k => [k, $(k).type === 'checkbox' ? $(k).checked : $(k).value]));
// 'model' nie jest zwykłą kontrolką (opcje dochodzą z serwera), ale preset może go ustawić
const SETTABLE = [...ALL, 'model'];

const PRESETS = {
  portret:   { gamma: 0.9, contrast: 0.2, detail: 0.5, detail_radius: 8, detail_guard: 0.7,
               detail_clamp: 0.08, micro: 0.18, micro_radius: 2.5, median: '3', relief_mm: 8,
               base_mm: 3, shape: 'rect', trim: false, clip_low: 1, clip_high: 99.5 },
  detal:     { gamma: 0.85, contrast: 0.25, detail: 1.0, detail_radius: 5, detail_guard: 0.75,
               detail_clamp: 0.13, micro: 0.4, micro_radius: 1.8, median: '3', bilateral: 3,
               relief_mm: 10, resolution: 800, exres: 2400 },
  logo:      { gamma: 1, contrast: 0.9, detail: 0.2, detail_radius: 3, micro: 0, floor: 0.35,
               floor_soft: 0.02, median: '3', relief_mm: 3, base_mm: 2, edge_falloff: 0 },
  krajobraz: { gamma: 1.2, contrast: 0.1, detail: 0.45, detail_radius: 12, detail_guard: 0.4,
               micro: 0.12, relief_mm: 5, base_mm: 2, clip_low: 1, clip_high: 99 },
  medalion:  { shape: 'ellipse', trim: true, margin: 0.05, edge_falloff: 0.12, gamma: 0.9,
               contrast: 0.3, detail: 0.55, micro: 0.2, relief_mm: 7, base_mm: 3, width_mm: 80 },
  wycinanka: { alpha_cut: true, trim: true, alpha_threshold: 0.5, alpha_grow: -1,
               min_island: 0.3, shape: 'rect', margin: 0, edge_falloff: 0, floor: 0,
               gamma: 0.9, contrast: 0.25, detail: 0.5, micro: 0.2, median: '3',
               relief_mm: 8, base_mm: 2 },
  litofania: { invert: true, gamma: 1, contrast: 0, detail: 0.25, micro: 0.5, micro_radius: 1.5,
               relief_mm: 2.5, base_mm: 0.6, solid: true, median: '0', clip_low: 0, clip_high: 100 },
  brelok:    { model: 'dav2-large', input_size: 1278, tiles: '4', tile_blend: 1.0,
               invert: false, clip_low: 4.3, clip_high: 100, gamma: 2.42, contrast: 1.0,
               median: '5', bilateral: 5, smooth: 0, detail: 1.85, detail_radius: 16.5,
               detail_guard: 1.0, detail_clamp: 0.135, micro: 0.18, micro_radius: 1.1,
               floor: 0.14, floor_soft: 0.175, edge_falloff: 0.30, shape: 'rect',
               corner: 0.15, margin: 0.06, trim: true,
               alpha_cut: false, alpha_threshold: 0.11, alpha_grow: 0,
               cut_level: 0.01, min_island: 1.7,
               resolution: 560, width_mm: 100, relief_mm: 14.3, base_mm: 0.0,
               solid: true, exres: 1200 },
};

function apply(vals) {
  let needsNet = false;
  for (const [k, v] of Object.entries(vals)) {
    const el = $(k);
    if (!el || !SETTABLE.includes(k)) continue;
    const before = el.type === 'checkbox' ? el.checked : el.value;
    if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
    const after = el.type === 'checkbox' ? el.checked : el.value;
    label(k);
    if (String(before) !== String(after) && (DEPTH_ONLY.has(k) || k === 'model')) needsNet = true;
  }
  save();
  if ($('model')) localStorage.setItem('depthgen_model', $('model').value);
  // preset zmieniający model albo kafle wymaga ponownego policzenia sieci,
  // sama przebudowa siatki tego nie załatwi
  if (needsNet) needsDepth(true);
  schedule(30);
}

function save() {
  const s = {};
  for (const k of ALL) s[k] = $(k).type === 'checkbox' ? $(k).checked : $(k).value;
  try { localStorage.setItem('depthgen', JSON.stringify(s)); } catch { /* brak miejsca */ }
}

function restore() {
  try {
    const s = JSON.parse(localStorage.getItem('depthgen') || 'null');
    if (s) for (const [k, v] of Object.entries(s)) {
      const el = $(k);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = !!v; else el.value = v;
      label(k);
    }
  } catch { /* ignoruj */ }
}

$('preset').addEventListener('change', e => {
  const p = PRESETS[e.target.value];
  if (!p) return;
  // Czyszczenie JPEG i upscaling zależą od konkretnego pliku, a nie od tego,
  // co z niego robimy — preset ich nie rusza, jeśli sam ich nie ustala.
  const base = { ...DEFAULTS };
  for (const k of PREP) if (!(k in p)) delete base[k];
  apply({ ...base, ...p });
});
$('reset').addEventListener('click', () => { $('preset').value = ''; apply(DEFAULTS); });

$('drop').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', e => e.target.files[0] && upload(e.target.files[0]));
for (const ev of ['dragenter', 'dragover']) {
  $('drop').addEventListener(ev, e => { e.preventDefault(); $('drop').classList.add('over'); });
}
for (const ev of ['dragleave', 'drop']) {
  $('drop').addEventListener(ev, e => { e.preventDefault(); $('drop').classList.remove('over'); });
}
$('drop').addEventListener('drop', e => {
  const f = e.dataTransfer.files[0];
  if (f) upload(f);
});
document.addEventListener('dragover', e => e.preventDefault());
document.addEventListener('drop', e => {
  e.preventDefault();
  const f = e.dataTransfer?.files?.[0];
  if (f && f.type.startsWith('image/')) upload(f);
});

$('depthFile').addEventListener('change', async e => {
  const f = e.target.files[0];
  if (!f || !state.id) return;
  const fd = new FormData();
  fd.append('id', state.id);
  fd.append('file', f);
  busy(true, 'Wczytuję mapę głębi…');
  const r = await fetch('/api/depth-upload', { method: 'POST', body: fd });
  busy(false);
  if (r.ok) {
    state.hasDepth = true;
    $('export').disabled = false;
    $('saveDepth').disabled = false;
    $('hint').classList.add('hidden');
    await rebuild();
    viewer.fit();
  } else $('prog').textContent = 'Błąd wczytywania mapy głębi.';
});

$('gen').addEventListener('click', generate);
$('export').addEventListener('click', doExport);
$('saveDepth').addEventListener('click', saveHeight);
$('refresh').addEventListener('click', rebuild);
$('fit').addEventListener('click', () => viewer.fit());
$('front').addEventListener('click', () => viewer.fit(true));
$('wire').addEventListener('change', e => viewer.setWireframe(e.target.checked));
$('spin').addEventListener('change', e => viewer.spin = e.target.checked);
$('matsel').addEventListener('change', e => viewer.setMaterial(e.target.value));
$('lightrot').addEventListener('input', e => viewer.setLight(+e.target.value));
document.querySelectorAll('.grp>h3').forEach(h =>
  h.addEventListener('click', () => h.parentElement.classList.toggle('open')));

viewer.setLight(45);

// ---------- start ----------
fetch('/api/info').then(r => r.json()).then(j => {
  $('hw').textContent = `${j.device.name} · ${j.device.device.toUpperCase()} · torch ${j.device.torch}`;
  const sel = $('model');
  sel.innerHTML = '';
  for (const m of j.models) {
    const o = document.createElement('option');
    o.value = m.key; o.textContent = m.label; o.dataset.size = m.default_size;
    sel.appendChild(o);
  }
  const sr = $('sr_model');
  sr.innerHTML = '';
  for (const m of j.sr_models || []) {
    const o = document.createElement('option');
    o.value = m.key; o.textContent = m.label;
    sr.appendChild(o);
  }
  sel.value = localStorage.getItem('depthgen_model') || 'dav2-large';
  sel.addEventListener('change', () => {
    localStorage.setItem('depthgen_model', sel.value);
    const s = sel.selectedOptions[0].dataset.size;
    if (s && !localStorage.getItem('depthgen')) { $('input_size').value = s; label('input_size'); }
  });
  restore();
});
