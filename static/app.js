(() => {
  const state = { admin: Boolean(window.COVALT && window.COVALT.admin), adminToken: null, polling: null };
  const $ = (selector) => document.querySelector(selector);
  const form = $('#generateForm');
  const generateButton = $('#generateButton');
  const progressCard = $('#progressCard');
  const progressBar = $('#progressBar');
  const progressPercent = $('#progressPercent');
  const progressMessage = $('#progressMessage');
  const progressDetail = $('#progressDetail');
  const duration = $('#duration');
  const durationValue = $('#durationValue');
  const videoGrid = $('#videoGrid');
  const toastNode = $('#toast');
  const chatMessages = $('#chatMessages');
  const MEMORY_KEY = 'covalt_text_memory_v1';
  let chatHistory = [];
  try { chatHistory = JSON.parse(localStorage.getItem(MEMORY_KEY) || '[]').filter((item) => item.role && item.content).slice(-20); } catch { chatHistory = []; }

  const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
  const formatDuration = (seconds) => {
    const total = Math.round(Number(seconds) || 0);
    if (total < 60) return `${total} сек`;
    return `${Math.floor(total / 60)} мин ${String(total % 60).padStart(2, '0')} сек`;
  };
  const formatDate = (value) => {
    try { return new Intl.DateTimeFormat('ru-RU', { day: '2-digit', month: 'short', year: 'numeric' }).format(new Date(value)); }
    catch { return 'сегодня'; }
  };
  function saveMemory() {
    try { localStorage.setItem(MEMORY_KEY, JSON.stringify(chatHistory.slice(-20))); } catch { /* storage can be disabled */ }
    $('#memoryStatus').textContent = `Память: ${chatHistory.length} сообщений`;
  }
  function restoreMemory() {
    chatHistory.forEach((item) => appendChatMessage(item.role, item.content));
    saveMemory();
  }
  let toastTimer;
  const toast = (message) => {
    toastNode.textContent = message;
    toastNode.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastNode.classList.remove('show'), 3600);
  };
  const api = async (url, options = {}) => {
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    if (state.adminToken) headers['X-Covalt-Admin'] = state.adminToken;
    const response = await fetch(url, { credentials: 'include', headers, ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'Что-то пошло не так');
    return data;
  };

  function videoCard(video, reviewMode = false) {
    const actions = reviewMode
      ? `<button class="publish-button" data-action="publish" data-id="${escapeHtml(video.id)}">Опубликовать ↗</button><button class="delete-button" data-action="delete" data-id="${escapeHtml(video.id)}">Удалить</button>`
      : `<a class="download-button" href="${escapeHtml(video.download_url)}">Скачать MP4 ↓</a>`;
    return `<article class="video-card">
      <div class="video-frame"><video controls preload="metadata" playsinline poster="${escapeHtml(video.thumbnail_url || '')}" src="${escapeHtml(video.video_url || '')}"></video></div>
      <div class="video-info"><div class="video-top"><h3 class="video-title" title="${escapeHtml(video.title)}">${escapeHtml(video.title)}</h3><span class="video-duration">${formatDuration(video.duration)}</span></div>
      <p class="video-prompt">${escapeHtml(video.prompt)}</p><div class="video-footer"><span class="video-date">${reviewMode ? '<span class="draft-badge">DRAFT</span>' : formatDate(video.created_at)}</span><span>${actions}</span></div></div>
    </article>`;
  }

  function renderVideos(videos) {
    const published = videos.filter((video) => video.published);
    const drafts = videos.filter((video) => !video.published);
    $('#libraryCount').textContent = `${published.length} ${published.length === 1 ? 'работа' : 'работ'}`;
    videoGrid.innerHTML = published.length
      ? published.map((video) => videoCard(video)).join('')
      : '<div class="empty-state"><span class="empty-ring"></span><p>Здесь появятся опубликованные сцены.</p><small>Сгенерируйте первое видео выше.</small></div>';

    const review = $('#adminReview');
    review.hidden = !state.admin;
    if (state.admin) {
      $('#reviewCount').textContent = `${drafts.length} ${drafts.length === 1 ? 'черновик' : 'черновиков'}`;
      $('#reviewGrid').innerHTML = drafts.map((video) => videoCard(video, true)).join('');
    }
  }

  async function refreshVideos() {
    try {
      const data = await api('/api/videos');
      state.admin = Boolean(data.admin);
      renderVideos(data.videos || []);
      updateAdminUi();
    } catch (error) { toast(error.message); }
  }

  function updateAdminUi() {
    $('#adminLabel').textContent = state.admin ? 'Администратор' : 'Вход админа';
    $('#adminButton').classList.toggle('active', state.admin);
    $('#adminButton .lock-icon').textContent = state.admin ? '✓' : '⌑';
    $('#adminBanner').hidden = !state.admin;
    $('#newsEditor').hidden = !state.admin;
  }

  const modeCaptions = {
    video: 'Локальный генератор движения · обучаемая MLP · MP4 до 5 минут',
    chat: 'Память Covalt · разбор смысла · опциональный поиск по интернету',
    image: 'Локальный Canvas · понимание промпта · PNG-рендер',
  };
  function setMode(mode) {
    if (mode !== 'chat') {
      toast('Режим видео/картинок временно выключен: сначала тестируем понимание текста.');
      return;
    }
    document.querySelectorAll('.mode-tab').forEach((button) => button.classList.toggle('active', button.dataset.mode === mode));
    $('#videoWorkspace').hidden = mode !== 'video';
    $('#progressCard').hidden = mode !== 'video' || !$('#progressCard').dataset.running;
    $('#chatPanel').hidden = mode !== 'chat';
    $('#imagePanel').hidden = mode !== 'image';
    $('#modeCaption').textContent = modeCaptions[mode] || modeCaptions.video;
  }
  document.querySelectorAll('.mode-tab').forEach((button) => button.addEventListener('click', () => setMode(button.dataset.mode)));

  function appendChatMessage(role, content, sources = [], understanding = null, meta = null) {
    const isAssistant = role === 'assistant';
    const sourceHtml = sources.length ? `<div class="source-list">${sources.map((source) => `<a href="${escapeHtml(source.url)}" target="_blank" rel="noopener"><b>${escapeHtml(source.title)}</b><small>${escapeHtml(source.snippet || source.url)}</small></a>`).join('')}</div>` : '';
    const browserSearchHtml = meta && meta.search_links && meta.search_links.length ? `<div class="browser-search-links"><span>Сервер не получил ответ. Открыть запрос в браузере:</span>${meta.search_links.map((link) => `<a href="${escapeHtml(link.url)}" target="_blank" rel="noopener">${escapeHtml(link.title)} ↗</a>`).join('')}</div>` : '';
    const searchHtml = meta && meta.search_status === 'server-network-unavailable' ? `<div class="search-status">Поиск включён, но сервер не получил источники. Covalt не будет их выдумывать.</div>${browserSearchHtml}` : '';
    const intentHtml = understanding ? `<div class="intent-chips"><span>${escapeHtml(understanding.scene)}</span><span>${escapeHtml(understanding.action)}</span><span>${escapeHtml(understanding.palette)}</span><span>${Math.round((understanding.confidence || 0) * 100)}% match</span></div>` : '';
    const timeHtml = meta && meta.thinking_ms ? `<small class="thinking-meta">анализ и составление · ${escapeHtml(meta.thinking_ms)} ms</small>` : '';
    const node = document.createElement('div');
    node.className = `chat-message ${isAssistant ? 'assistant-message' : 'user-message'}`;
    node.innerHTML = isAssistant
      ? `<span class="message-avatar">C</span><div><b>Covalt</b><p>${escapeHtml(content).replace(/\n/g, '<br>')}</p>${timeHtml}${intentHtml}${searchHtml}${sourceHtml}</div>`
      : `<div><b>Вы</b><p>${escapeHtml(content).replace(/\n/g, '<br>')}</p></div>`;
    chatMessages.appendChild(node);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  $('#chatForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const input = $('#chatInput');
    const message = input.value.trim();
    if (!message) return;
    const useWeb = $('#chatWeb').checked;
    appendChatMessage('user', message);
    chatHistory.push({ role: 'user', content: message });
    saveMemory();
    input.value = '';
    const button = event.submitter || event.target.querySelector('button');
    const status = $('#chatStatus');
    const statusText = $('#chatStatusText');
    const stages = ['Covalt разбирает запрос…', 'Covalt проверяет контекст…', useWeb ? 'Covalt проверяет поиск…' : 'Covalt составляет ответ…'];
    let stage = 0;
    status.hidden = false;
    statusText.textContent = stages[stage];
    const thinkingNode = document.createElement('div');
    thinkingNode.className = 'chat-thinking';
    thinkingNode.innerHTML = '<span class="spinner"></span><span></span>';
    thinkingNode.querySelector('span:last-child').textContent = stages[stage];
    chatMessages.appendChild(thinkingNode);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    const stageTimer = setInterval(() => { stage = Math.min(stage + 1, stages.length - 1); statusText.textContent = stages[stage]; thinkingNode.querySelector('span:last-child').textContent = stages[stage]; }, 500);
    button.disabled = true;
    button.querySelector('span').textContent = 'Ждём ответ…';
    try {
      const result = await api('/api/chat', { method: 'POST', body: JSON.stringify({ message, history: chatHistory.slice(-8), use_web: useWeb }) });
      appendChatMessage('assistant', result.answer, result.sources || [], result.understanding, result);
      chatHistory.push({ role: 'assistant', content: result.answer });
      saveMemory();
    } catch (error) { appendChatMessage('assistant', `Не получилось выполнить запрос: ${error.message}`); }
    clearInterval(stageTimer);
    thinkingNode.remove();
    status.hidden = true;
    button.disabled = false;
    button.querySelector('span').textContent = 'Отправить';
  });

  $('#imageForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const button = $('#imageButton');
    button.disabled = true;
    button.querySelector('span').textContent = 'Рендерим…';
    try {
      const result = await api('/api/image', { method: 'POST', body: JSON.stringify({ prompt: $('#imagePrompt').value }) });
      const intent = result.understanding || {};
      $('#imageUnderstanding').textContent = `Covalt понял: ${intent.scene || 'scene'} · ${intent.action || 'motion'} · ${intent.palette || 'palette'} · ${Math.round((intent.confidence || 0) * 100)}% match`;
      $('#imageResult').innerHTML = `<div class="generated-image"><img src="${escapeHtml(result.url)}" alt="${escapeHtml(result.prompt)}"><div class="image-result-footer"><span>${escapeHtml(result.scene)} / ${escapeHtml(result.palette || '')}</span><a class="download-button" href="${escapeHtml(result.url)}" download="covalt-${escapeHtml(result.id)}.png">Скачать PNG ↓</a></div></div>`;
      toast('Изображение Covalt готово.');
    } catch (error) { toast(error.message); }
    button.disabled = false;
    button.querySelector('span').textContent = 'Создать изображение';
  });

  async function loadNews() {
    try {
      const data = await api('/api/news');
      const items = data.news || [];
      $('#newsGrid').innerHTML = items.length ? items.map((item) => `<article class="news-card"><div class="news-card-top"><span class="news-number">${String(items.indexOf(item) + 1).padStart(2, '0')}</span><span>${formatDate(item.updated_at || item.created_at)}</span></div><h3>${escapeHtml(item.title)}</h3><p>${escapeHtml(item.body).replace(/\n/g, '<br>')}</p>${state.admin && !item.published ? '<span class="draft-badge">DRAFT</span>' : ''}</article>`).join('') : '<div class="news-empty">Новости появятся здесь, когда команда Covalt опубликует первый апдейт.</div>';
      if (state.admin) {
        $('#adminNewsList').innerHTML = items.length ? items.map((item) => `<div class="admin-news-row"><div><b>${escapeHtml(item.title)}</b><small>${item.published ? 'Опубликовано' : 'Черновик'} · ${formatDate(item.updated_at || item.created_at)}</small></div><span><button class="publish-button" data-news-action="edit" data-news-id="${escapeHtml(item.id)}">Изменить</button><button class="delete-button" data-news-action="delete" data-news-id="${escapeHtml(item.id)}">Удалить</button></span></div>`).join('') : '<small class="news-empty">Список новостей пуст.</small>';
        $('#adminNewsList').dataset.items = JSON.stringify(items);
      }
    } catch (error) { toast(error.message); }
  }

  duration.addEventListener('input', () => { durationValue.textContent = duration.value; });
  document.querySelectorAll('[data-prompt]').forEach((button) => button.addEventListener('click', () => { $('#prompt').value = button.dataset.prompt; $('#prompt').focus(); }));
  $('#refreshButton').addEventListener('click', refreshVideos);

  async function pollJob(jobId) {
    try {
      const job = await api(`/api/jobs/${encodeURIComponent(jobId)}`);
      const percent = Math.max(0, Math.min(100, Math.round((job.progress || 0) * 100)));
      progressBar.style.width = `${percent}%`;
      progressPercent.textContent = `${percent}%`;
      progressMessage.textContent = job.message || 'Работаем…';
      if (job.status === 'done') {
        progressDetail.textContent = 'Готово. Ролик сохранён как MP4.';
        delete progressCard.dataset.running;
        generateButton.disabled = false;
        generateButton.querySelector('span').textContent = 'Сгенерировать ещё';
        if (state.admin) toast('Видео готово — оно ждёт публикации.'); else toast('Видео готово — войдите как админ, чтобы его опубликовать.');
        await refreshVideos();
        setTimeout(() => { progressCard.hidden = true; }, 4500);
        return;
      }
      if (job.status === 'error') throw new Error(job.message || 'Генерация не удалась');
      state.polling = setTimeout(() => pollJob(jobId), 850);
    } catch (error) {
      clearTimeout(state.polling);
      progressCard.hidden = true;
      delete progressCard.dataset.running;
      generateButton.disabled = false;
      generateButton.querySelector('span').textContent = 'Сгенерировать видео';
      toast(error.message);
    }
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();
    generateButton.disabled = true;
    generateButton.querySelector('span').textContent = 'Генерация…';
    progressCard.hidden = false;
    progressCard.dataset.running = '1';
    progressBar.style.width = '0%'; progressPercent.textContent = '0%';
    progressMessage.textContent = 'Запускаем движок…';
    progressDetail.textContent = 'Нейросеть готовит сцену. Для длинного ролика это может занять немного времени.';
    try {
      const data = await api('/api/generate', { method: 'POST', body: JSON.stringify({ prompt: $('#prompt').value, title: $('#title').value, duration: Number(duration.value) }) });
      pollJob(data.job_id);
    } catch (error) {
      generateButton.disabled = false; generateButton.querySelector('span').textContent = 'Сгенерировать видео'; progressCard.hidden = true; toast(error.message);
    }
  });

  const modal = $('#loginModal');
  $('#adminButton').addEventListener('click', () => {
    if (state.admin) { document.querySelector('#adminReview').scrollIntoView({ behavior: 'smooth' }); return; }
    modal.hidden = false; setTimeout(() => $('#password').focus(), 20);
  });
  $('#closeModal').addEventListener('click', () => { modal.hidden = true; });
  modal.addEventListener('click', (event) => { if (event.target === modal) modal.hidden = true; });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') modal.hidden = true; });
  $('#loginForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    $('#loginError').textContent = '';
    try {
      const loginData = await api('/api/login', { method: 'POST', body: JSON.stringify({ password: $('#password').value }) });
      state.admin = true;
      state.adminToken = loginData.admin_token || null;
      modal.hidden = true;
      $('#password').value = '';
      // The login response already contains the newest queue. Render it now;
      // do not wait for another request before showing the admin's drafts.
      renderVideos(loginData.videos || []);
      updateAdminUi();
      await loadNews();
      toast('Режим администратора включён.');
    } catch (error) { $('#loginError').textContent = error.message; }
  });
  $('#logoutButton').addEventListener('click', async () => { try { await api('/api/logout', { method: 'POST' }); state.admin = false; state.adminToken = null; updateAdminUi(); await refreshVideos(); await loadNews(); toast('Вы вышли из режима администратора.'); } catch (error) { toast(error.message); } });

  $('#newNewsButton').addEventListener('click', () => {
    $('#newsId').value = '';
    $('#newsTitle').value = '';
    $('#newsBody').value = '';
    $('#newsPublished').checked = true;
    $('#newsSubmitLabel').textContent = 'Опубликовать новость';
    $('#newsTitle').focus();
  });

  $('#newsForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    const id = $('#newsId').value.trim();
    const payload = { title: $('#newsTitle').value, body: $('#newsBody').value, published: $('#newsPublished').checked };
    try {
      await api(id ? `/api/news/${encodeURIComponent(id)}` : '/api/news', { method: id ? 'PUT' : 'POST', body: JSON.stringify(payload) });
      $('#newsFormStatus').textContent = 'Сохранено';
      toast(id ? 'Новость обновлена.' : 'Новость опубликована.');
      await loadNews();
      setTimeout(() => { $('#newsFormStatus').textContent = ''; }, 2000);
    } catch (error) { $('#newsFormStatus').textContent = error.message; }
  });

  document.addEventListener('click', async (event) => {
    const newsButton = event.target.closest('[data-news-action]');
    if (newsButton) {
      const items = JSON.parse($('#adminNewsList').dataset.items || '[]');
      const item = items.find((entry) => entry.id === newsButton.dataset.newsId);
      if (!item) return;
      if (newsButton.dataset.newsAction === 'edit') {
        $('#newsId').value = item.id;
        $('#newsTitle').value = item.title;
        $('#newsBody').value = item.body;
        $('#newsPublished').checked = Boolean(item.published);
        $('#newsSubmitLabel').textContent = 'Сохранить изменения';
        $('#newsTitle').focus();
      } else if (newsButton.dataset.newsAction === 'delete' && window.confirm('Удалить эту новость?')) {
        try { await api(`/api/news/${encodeURIComponent(item.id)}`, { method: 'DELETE' }); toast('Новость удалена.'); await loadNews(); }
        catch (error) { toast(error.message); }
      }
      return;
    }
    const button = event.target.closest('[data-action]');
    if (!button || !button.closest('#reviewGrid')) return;
    const id = button.dataset.id;
    button.disabled = true;
    try {
      if (button.dataset.action === 'publish') { await api(`/api/videos/${encodeURIComponent(id)}/publish`, { method: 'POST' }); toast('Видео опубликовано в библиотеке.'); }
      if (button.dataset.action === 'delete') { if (!window.confirm('Удалить это видео навсегда?')) { button.disabled = false; return; } await api(`/api/videos/${encodeURIComponent(id)}`, { method: 'DELETE' }); toast('Видео удалено.'); }
      await refreshVideos();
    } catch (error) { button.disabled = false; toast(error.message); }
  });

  $('#clearMemory').addEventListener('click', () => {
    chatHistory = [];
    try { localStorage.removeItem(MEMORY_KEY); } catch { /* storage can be disabled */ }
    $('#chatMessages').querySelectorAll('.user-message, .assistant-message:not(:first-child)').forEach((node) => node.remove());
    saveMemory();
    toast('Память текущего диалога очищена.');
  });
  restoreMemory();
  setMode('chat');
  updateAdminUi();
  refreshVideos();
  loadNews();
})();
