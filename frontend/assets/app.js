/**
 * ═══════════════════════════════════════════════════════════════════════════
 * ERC Analyzer — Enterprise Frontend Application
 * Hybrid Retrieval · Cross-Encoder Reranking · Deterministic Anti-Hallucination
 * ═══════════════════════════════════════════════════════════════════════════
 */

// ── Application State ────────────────────────────────────────────────────────
const state = {
  documents: [],
  selectedDocs: new Set(),
  messages: [],
  loading: false,
  health: { status: 'checking', database: 'unknown' }
};

// ── DOM References ───────────────────────────────────────────────────────────
const $ = (id) => document.getElementById(id);
const chatArea          = $('chat-area');
const docList           = $('doc-list');
const docCountBadge     = $('doc-count-badge');
const uploadZone        = $('upload-zone');
const uploadInput       = $('upload-input');
const uploadProgress    = $('upload-progress');
const progressFilename  = $('progress-filename');
const progressPct       = $('progress-pct');
const progressFill      = $('progress-fill');
const progressStage     = $('progress-stage');
const queryInput        = $('query-input');
const sendBtn           = $('send-btn');
const welcomeState      = $('welcome-state');
const toastContainer    = $('toast-container');
const scopeSummary      = $('scope-summary');
const scopeClearBtn     = $('scope-clear-btn');
const filterBanner      = $('filter-active-banner');
const filterDocCount    = $('filter-doc-count');
const dbDot             = $('db-dot');
const dbVal             = $('db-val');
const btnClearChat      = $('btn-clear-chat');

// ── Toast Notification System ────────────────────────────────────────────────
function showToast(message, type = 'info', duration = 4000) {
  const toast = document.createElement('div');
  toast.className = `toast-item ${type}`;

  const iconMap = {
    success: '✓',
    error: '✕',
    info: 'ℹ',
    warning: '⚠️'
  };

  toast.innerHTML = `
    <span class="toast-icon">${iconMap[type] || 'ℹ'}</span>
    <span class="toast-msg">${escapeHtml(message)}</span>
  `;

  toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.classList.add('hide');
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ── API Fetch Client ─────────────────────────────────────────────────────────
async function apiRequest(endpoint, options = {}) {
  const response = await fetch(endpoint, options);
  const contentType = response.headers.get('content-type') || '';
  const isJson = contentType.includes('application/json');
  const payload = isJson ? await response.json() : await response.text();

  if (!response.ok) {
    let errorMsg = 'An error occurred during request.';
    if (payload && payload.detail) {
      if (typeof payload.detail === 'string') {
        errorMsg = payload.detail;
      } else if (payload.detail.reason) {
        errorMsg = payload.detail.reason;
      } else {
        errorMsg = JSON.stringify(payload.detail);
      }
    } else if (typeof payload === 'string') {
      errorMsg = payload;
    }
    const err = new Error(errorMsg);
    err.status = response.status;
    err.payload = payload;
    throw err;
  }

  return payload;
}

// ── Health Telemetry Check ───────────────────────────────────────────────────
async function checkHealth() {
  try {
    const data = await apiRequest('/health');
    state.health = data;
    if (data.database === 'connected') {
      dbDot.className = 'pulse-dot green';
      dbVal.textContent = 'Connected (pgvector)';
      dbVal.style.color = 'var(--success)';
    } else {
      dbDot.className = 'pulse-dot red';
      dbVal.textContent = data.database || 'Degraded';
      dbVal.style.color = 'var(--danger)';
    }
  } catch (err) {
    dbDot.className = 'pulse-dot red';
    dbVal.textContent = 'Offline';
    dbVal.style.color = 'var(--danger)';
  }
}

// ── Document Vault Operations ────────────────────────────────────────────────
async function loadDocuments() {
  try {
    const data = await apiRequest('/documents');
    state.documents = data.documents || [];
    renderVaultList();
    updateScopeUI();
  } catch (err) {
    console.warn('Failed to fetch documents:', err);
  }
}

function renderVaultList() {
  docCountBadge.textContent = `${state.documents.length} Docs`;

  if (state.documents.length === 0) {
    docList.innerHTML = `
      <div class="vault-empty-state">
        <div class="empty-icon-wrap">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="10"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>
        </div>
        <h4>No Documents Ingested</h4>
        <p>Upload regulatory PDFs (GDPR, EU AI Act, HIPAA) to start asking compliance questions.</p>
      </div>
    `;
    return;
  }

  docList.innerHTML = state.documents.map((doc) => {
    const isSelected = state.selectedDocs.has(doc.file_name);
    const statusClass = (doc.status || 'completed').toLowerCase();
    const createdDate = doc.created_at ? formatDate(doc.created_at) : 'Just now';
    const pagesLabel = doc.page_count ? `${doc.page_count} pages` : 'Indexed';

    return `
      <div class="doc-card ${isSelected ? 'selected' : ''}" data-name="${escapeHtml(doc.file_name)}" onclick="toggleDocSelection('${escapeHtml(doc.file_name)}')">
        <input 
          type="checkbox" 
          class="doc-checkbox" 
          ${isSelected ? 'checked' : ''} 
          onclick="event.stopPropagation(); toggleDocSelection('${escapeHtml(doc.file_name)}')"
        />
        <div class="doc-icon-wrap">📄</div>
        <div class="doc-details">
          <div class="doc-filename" title="${escapeHtml(doc.file_name)}">${escapeHtml(doc.file_name)}</div>
          <div class="doc-metadata">
            <span>${pagesLabel}</span>
            <span>•</span>
            <span>${createdDate}</span>
            <span>•</span>
            <span class="doc-status-badge ${statusClass}">${doc.status || 'READY'}</span>
          </div>
        </div>
        <button 
          class="doc-delete-btn" 
          title="Delete document and embeddings" 
          onclick="deleteDocument(event, '${doc.document_id}', '${escapeHtml(doc.file_name)}')"
        >
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg>
        </button>
      </div>
    `;
  }).join('');
}

function toggleDocSelection(fileName) {
  if (state.selectedDocs.has(fileName)) {
    state.selectedDocs.delete(fileName);
  } else {
    state.selectedDocs.add(fileName);
  }
  renderVaultList();
  updateScopeUI();
}

function clearDocSelection() {
  state.selectedDocs.clear();
  renderVaultList();
  updateScopeUI();
}

function updateScopeUI() {
  const count = state.selectedDocs.size;
  if (count > 0) {
    scopeSummary.textContent = `Scoping to ${count} selected document(s)`;
    scopeClearBtn.style.display = 'inline-block';
    filterBanner.style.display = 'flex';
    filterDocCount.textContent = count;
  } else {
    scopeSummary.textContent = 'Querying entire vault (all documents)';
    scopeClearBtn.style.display = 'none';
    filterBanner.style.display = 'none';
  }
}

async function deleteDocument(event, docId, fileName) {
  event.stopPropagation();
  if (!confirm(`Are you sure you want to delete "${fileName}" and all of its vector embeddings from the corpus?`)) {
    return;
  }

  try {
    await apiRequest(`/documents/${docId}`, { method: 'DELETE' });
    state.selectedDocs.delete(fileName);
    showToast(`"${fileName}" removed successfully`, 'success');
    await loadDocuments();
  } catch (err) {
    showToast(`Failed to delete document: ${err.message}`, 'error');
  }
}

// ── File Upload Handler ──────────────────────────────────────────────────────
uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const files = Array.from(e.dataTransfer.files).filter((f) => f.name.toLowerCase().endsWith('.pdf'));
  if (files.length > 0) {
    uploadBatchFiles(files);
  } else {
    showToast('Please upload valid PDF files (.pdf)', 'warning');
  }
});

uploadInput.addEventListener('change', () => {
  const files = Array.from(uploadInput.files);
  if (files.length > 0) {
    uploadBatchFiles(files);
  }
  uploadInput.value = '';
});

async function uploadBatchFiles(files) {
  for (const file of files) {
    await uploadSingleFile(file);
  }
}

async function uploadSingleFile(file) {
  uploadProgress.classList.add('visible');
  progressFilename.textContent = file.name;
  progressPct.textContent = '0%';
  progressFill.style.width = '0%';
  progressStage.textContent = 'Hashing SHA-256 for deduplication…';

  let currentPct = 10;
  progressFill.style.width = `${currentPct}%`;
  progressPct.textContent = `${currentPct}%`;

  const stages = [
    'Parsing PDF text and page structure…',
    'Recursive chunking (1200 chars / 200 overlap)…',
    'Generating 768-dim embeddings with Gemini…',
    'Writing HNSW vectors to PostgreSQL…'
  ];

  let stageIndex = 0;
  const timer = setInterval(() => {
    if (stageIndex < stages.length) {
      progressStage.textContent = stages[stageIndex];
      stageIndex++;
    }
    currentPct = Math.min(currentPct + 18, 88);
    progressFill.style.width = `${currentPct}%`;
    progressPct.textContent = `${currentPct}%`;
  }, 450);

  try {
    const formData = new FormData();
    formData.append('file', file);

    const result = await apiRequest('/documents/ingest', {
      method: 'POST',
      body: formData,
    });

    clearInterval(timer);
    progressFill.style.width = '100%';
    progressPct.textContent = '100%';
    progressStage.textContent = 'Ingestion complete!';

    if (result.status === 'ALREADY_INGESTED') {
      showToast(`"${file.name}" was already ingested (identical SHA-256)`, 'info');
    } else {
      showToast(`Ingested "${file.name}" — ${result.chunk_count || 0} chunks indexed`, 'success');
    }

    await loadDocuments();
  } catch (err) {
    clearInterval(timer);
    progressStage.textContent = 'Ingestion failed';
    showToast(`Upload failed: ${err.message}`, 'error');
  } finally {
    setTimeout(() => {
      uploadProgress.classList.remove('visible');
      progressFill.style.width = '0%';
    }, 1800);
  }
}

// ── Compliance Q&A Chat Pipeline ─────────────────────────────────────────────
queryInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) {
    e.preventDefault();
    submitQuery();
  }
});

queryInput.addEventListener('input', () => {
  queryInput.style.height = 'auto';
  queryInput.style.height = `${Math.min(queryInput.scrollHeight, 160)}px`;
  sendBtn.disabled = !queryInput.value.trim() || state.loading;
});

sendBtn.addEventListener('click', submitQuery);

if (btnClearChat) {
  btnClearChat.addEventListener('click', () => {
    chatArea.innerHTML = '';
    if (welcomeState) {
      chatArea.appendChild(welcomeState);
      welcomeState.style.display = 'block';
    }
    state.messages = [];
    showToast('Chat history cleared', 'info');
  });
}

function useSuggestion(text) {
  queryInput.value = text;
  queryInput.dispatchEvent(new Event('input'));
  queryInput.focus();
}

async function submitQuery() {
  const query = queryInput.value.trim();
  if (!query || state.loading) return;

  if (state.documents.length === 0) {
    showToast('Please upload at least one regulatory PDF document first', 'warning');
    return;
  }

  // Hide welcome hero
  if (welcomeState) {
    welcomeState.style.display = 'none';
  }

  // Render User Message
  appendUserMessage(query);
  queryInput.value = '';
  queryInput.style.height = 'auto';

  // Render Thinking Stepper
  const startTime = Date.now();
  const thinkingElement = appendThinkingStepper();
  state.loading = true;
  sendBtn.disabled = true;

  const docFilter = state.selectedDocs.size > 0 ? Array.from(state.selectedDocs) : null;

  try {
    const response = await apiRequest('/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        document_filter: docFilter,
      }),
    });

    const elapsedSeconds = ((Date.now() - startTime) / 1000).toFixed(1);
    thinkingElement.remove();
    appendAiMessage(response, elapsedSeconds);
  } catch (err) {
    thinkingElement.remove();
    appendErrorMessage(err);
  } finally {
    state.loading = false;
    sendBtn.disabled = !queryInput.value.trim();
    queryInput.focus();
  }
}

function appendUserMessage(text) {
  const msgRow = document.createElement('div');
  msgRow.className = 'chat-msg-row chat-msg-user';
  msgRow.innerHTML = `
    <div class="user-bubble">${escapeHtml(text)}</div>
    <div class="user-meta">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
  `;
  chatArea.appendChild(msgRow);
  scrollToBottom();
}

function appendThinkingStepper() {
  const card = document.createElement('div');
  card.className = 'chat-msg-row chat-msg-ai';
  card.innerHTML = `
    <div class="thinking-card">
      <div class="thinking-header">
        <div class="thinking-spinner"></div>
        <div class="thinking-title">Analyzing Regulatory Context…</div>
      </div>
      <div class="thinking-steps">
        <div class="step-item active" id="step-1">
          <div class="step-dot"></div>
          <span>Stage 1: HNSW Cosine (Dense) + BM25 (Sparse) Hybrid Retrieval</span>
        </div>
        <div class="step-item" id="step-2">
          <div class="step-dot"></div>
          <span>Stage 2: BGE Cross-Encoder Reranking (bge-reranker-v2-m3)</span>
        </div>
        <div class="step-item" id="step-3">
          <div class="step-dot"></div>
          <span>Stage 3: Deterministic Anti-Hallucination Guardrail Check</span>
        </div>
        <div class="step-item" id="step-4">
          <div class="step-dot"></div>
          <span>Stage 4: Gemini 3.8 Flash Grounded Synthesis</span>
        </div>
      </div>
    </div>
  `;
  chatArea.appendChild(card);
  scrollToBottom();

  // Progressively activate steps
  setTimeout(() => {
    const s2 = $('step-2');
    if (s2) s2.classList.add('active');
  }, 500);
  setTimeout(() => {
    const s3 = $('step-3');
    if (s3) s3.classList.add('active');
  }, 1200);
  setTimeout(() => {
    const s4 = $('step-4');
    if (s4) s4.classList.add('active');
  }, 1800);

  return card;
}

function appendAiMessage(result, elapsedSeconds) {
  const answer = result.answer || '';
  const citations = result.citations || [];
  const citationsVerified = result.citations_verified;

  const msgRow = document.createElement('div');
  msgRow.className = 'chat-msg-row chat-msg-ai';

  const citationsDrawerHtml = citations.length > 0 ? `
    <div class="citations-drawer">
      <div class="citations-header">
        <div class="citations-title">
          <span>📎 Grounding Citations</span>
          <span class="citations-count-badge">${citations.length} verified excerpt(s)</span>
        </div>
      </div>
      <div class="citations-grid">
        ${citations.map((c) => {
          const scorePercent = c.similarity_score != null ? Math.round(c.similarity_score * 100) : null;
          return `
            <div class="source-card">
              <div class="source-card-top">
                <span class="source-filename" title="${escapeHtml(c.document_name)}">📄 ${escapeHtml(c.document_name)}</span>
                <span class="source-page">Page ${c.page_number}</span>
              </div>
              ${scorePercent != null ? `
                <div class="source-match-bar" title="Cross-Encoder Relevance Score: ${scorePercent}%">
                  <div class="source-match-fill" style="width: ${Math.min(scorePercent, 100)}%;"></div>
                </div>
              ` : ''}
              ${c.excerpt ? `<div class="source-excerpt">"${escapeHtml(c.excerpt)}"</div>` : ''}
            </div>
          `;
        }).join('')}
      </div>
    </div>
  ` : '';

  msgRow.innerHTML = `
    <div class="ai-card">
      <div class="ai-card-header">
        <div class="ai-author">
          <div class="ai-avatar">AI</div>
          <span class="ai-name">ERC Compliance Engine</span>
          <span class="ai-badge">Gemini 3.8 Flash</span>
        </div>
        <div class="ai-header-meta">
          <span class="latency-tag">${elapsedSeconds}s</span>
          ${citationsVerified ? '<span class="verified-pill">🛡️ Citations Verified</span>' : ''}
        </div>
      </div>
      <div class="ai-card-body">
        ${formatAiAnswer(answer)}
      </div>
      ${citationsDrawerHtml}
    </div>
  `;

  chatArea.appendChild(msgRow);
  scrollToBottom();
}

function appendErrorMessage(err) {
  const msgRow = document.createElement('div');
  msgRow.className = 'chat-msg-row chat-msg-ai';

  let title = 'Query Processing Issue';
  let message = err.message || 'An unexpected error occurred.';
  let icon = '⚠️';

  if (err.status === 404) {
    icon = '🔍';
    title = 'Insufficient Regulatory Context (Guard 1 / Guard 2)';
    message = 'No retrieved document excerpts met the minimum relevance threshold (Cosine ≥ 0.50 or Cross-Encoder ≥ 0.0). The model strictly refrained from hallucinating an answer.';
  } else if (err.status === 422) {
    icon = '🛡️';
    title = 'Citation Verification Failed (Guard 3)';
    message = 'The generated compliance answer contained references that could not be validated against retrieved document chunks. The answer was rejected to preserve audit integrity.';
  }

  msgRow.innerHTML = `
    <div class="error-card">
      <span class="error-icon">${icon}</span>
      <div class="error-content">
        <strong>${escapeHtml(title)}</strong>
        <p>${escapeHtml(message)}</p>
      </div>
    </div>
  `;

  chatArea.appendChild(msgRow);
  scrollToBottom();
}

// ── Text Formatters & Markdown Helpers ───────────────────────────────────────
function formatAiAnswer(text) {
  if (!text) return '';

  // 1. Escape HTML
  let out = escapeHtml(text);

  // 2. Transform citations: [Document: ..., Page: X] into clickable interactive pills
  out = out.replace(
    /\[Document:\s*([^\]]+?),\s*Page:\s*(\d+)\]/gi,
    '<span class="citation-pill" title="Verified citation from $1, page $2">📄 $1 · p.$2</span>'
  );

  // 3. Bold text: **bold**
  out = out.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');

  // 4. Italic text: *italic*
  out = out.replace(/(?<!\*)\*(?!\*)(.*?)(?<!\*)\*(?!\*)/g, '<em>$1</em>');

  // 5. Code backticks: `code`
  out = out.replace(/`([^`]+)`/g, '<code style="background:rgba(255,255,255,0.08);padding:1px 5px;border-radius:4px;font-family:var(--font-mono);font-size:0.85em;">$1</code>');

  // 6. Split into paragraphs / list items
  const lines = out.split('\n');
  let inList = false;
  let result = [];

  for (const rawLine of lines) {
    const line = rawLine.trim();
    if (!line) {
      if (inList) {
        result.push('</ul>');
        inList = false;
      }
      continue;
    }

    if (line.startsWith('- ') || line.startsWith('• ') || line.startsWith('* ')) {
      if (!inList) {
        result.push('<ul>');
        inList = true;
      }
      result.push(`<li>${line.substring(2)}</li>`);
    } else if (/^\d+\.\s/.test(line)) {
      if (!inList) {
        result.push('<ol>');
        inList = true;
      }
      result.push(`<li>${line.replace(/^\d+\.\s/, '')}</li>`);
    } else {
      if (inList) {
        result.push('</ul>');
        inList = false;
      }
      result.push(`<p>${line}</p>`);
    }
  }

  if (inList) {
    result.push('</ul>');
  }

  return result.join('');
}

function escapeHtml(str) {
  return String(str || '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function formatDate(iso) {
  try {
    const date = new Date(iso);
    return date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' });
  } catch {
    return 'Recent';
  }
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

// ── Application Initialization ───────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  checkHealth();
  loadDocuments();

  // Periodic health & document list sync
  setInterval(checkHealth, 10000);
  setInterval(loadDocuments, 20000);
});
