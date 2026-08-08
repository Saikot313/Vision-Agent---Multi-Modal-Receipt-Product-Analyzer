const dropzone = document.getElementById('dropzone');
const fileInput = document.getElementById('fileInput');
const previewRow = document.getElementById('previewRow');
const previewImg = document.getElementById('previewImg');
const scanline = document.getElementById('scanline');
const fileChip = document.getElementById('fileChip');
const analyzeBtn = document.getElementById('analyzeBtn');
const status = document.getElementById('status');
const results = document.getElementById('results');

let selectedFile = null;

function setFile(file) {
  if (!file) return;
  if (!/^image\/(jpeg|png|jpg)$/.test(file.type)) {
    status.className = 'status err';
    status.textContent = 'Only JPG or PNG images are supported';
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  previewImg.src = url;
  fileChip.textContent = `${file.name} · ${(file.size / 1024).toFixed(1)} KB`;
  previewRow.style.display = 'flex';
  status.className = 'status';
  status.textContent = '';
  results.classList.remove('show');
}

dropzone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => setFile(e.target.files[0]));
['dragover', 'dragenter'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.add('drag'); })
);
['dragleave', 'drop'].forEach(ev =>
  dropzone.addEventListener(ev, e => { e.preventDefault(); dropzone.classList.remove('drag'); })
);
dropzone.addEventListener('drop', e => { e.preventDefault(); setFile(e.dataTransfer.files[0]); });

analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;
  analyzeBtn.disabled = true;
  scanline.classList.add('active');
  status.className = 'status';
  status.innerHTML = '<span class="spinner"></span> agent reasoning…';
  results.classList.remove('show');

  const form = new FormData();
  form.append('image', selectedFile);

  try {
    const res = await fetch('/analyze', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || 'Analysis failed');
    renderResult(data);
    status.textContent = '';
  } catch (err) {
    status.className = 'status err';
    status.textContent = `✕ ${err.message}`;
  } finally {
    analyzeBtn.disabled = false;
    scanline.classList.remove('active');
  }
});

function renderResult(data) {
  document.getElementById('typeBadge').textContent = data.image_type || 'unknown';
  document.getElementById('summaryText').textContent = data.summary || '';

  const objectBadges = document.getElementById('objectBadges');
  objectBadges.innerHTML = (data.detected_objects || [])
    .map(label => `<span class="badge">${escapeHtml(label)}</span>`)
    .join('');

  const priceCard = document.getElementById('priceCard');
  if (data.price_check) {
    priceCard.style.display = 'block';
    const pc = data.price_check;
    document.getElementById('computedTotal').textContent = pc.computed_total.toFixed(2);
    document.getElementById('printedTotal').textContent =
      pc.printed_total != null ? pc.printed_total.toFixed(2) : '—';
    const mismatchEl = document.getElementById('mismatchValue');
    mismatchEl.textContent = pc.mismatch ? 'Yes' : 'No';
    mismatchEl.className = 'price-value ' + (pc.mismatch ? 'mismatch-yes' : 'mismatch-no');
    document.getElementById('priceNote').textContent = pc.note || '';
  } else {
    priceCard.style.display = 'none';
  }

  const itemsCard = document.getElementById('itemsCard');
  const itemsBody = document.getElementById('itemsBody');
  if (data.line_items && data.line_items.length) {
    itemsCard.style.display = 'block';
    itemsBody.innerHTML = data.line_items.map(item => `
      <tr>
        <td>${escapeHtml(item.name)}</td>
        <td>${item.quantity}</td>
        <td>${item.unit_price != null ? item.unit_price.toFixed(2) : '—'}</td>
        <td>${item.line_total != null ? item.line_total.toFixed(2) : '—'}</td>
      </tr>
    `).join('');
  } else {
    itemsCard.style.display = 'none';
  }

  const ocrCard = document.getElementById('ocrCard');
  if (data.ocr_text) {
    ocrCard.style.display = 'block';
    document.getElementById('ocrText').textContent = data.ocr_text;
  } else {
    ocrCard.style.display = 'none';
  }

  const trace = data.agent_trace || [];
  document.getElementById('traceCount').textContent = `(${trace.length})`;
  document.getElementById('traceList').innerHTML = trace.map(t => `
    <div class="trace-item">
      <div class="trace-tool">${escapeHtml(t.tool)}</div>
      <div class="trace-args">${escapeHtml(JSON.stringify(t.arguments))}</div>
      <div class="trace-result">${escapeHtml(t.result_preview)}</div>
    </div>
  `).join('');

  results.classList.add('show');
}

const traceToggle = document.getElementById('traceToggle');
const traceList = document.getElementById('traceList');
const traceChevron = document.getElementById('traceChevron');
traceToggle.addEventListener('click', () => {
  const isOpen = traceList.style.display !== 'none';
  traceList.style.display = isOpen ? 'none' : 'flex';
  traceChevron.classList.toggle('open', !isOpen);
});

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str == null ? '' : String(str);
  return div.innerHTML;
}
