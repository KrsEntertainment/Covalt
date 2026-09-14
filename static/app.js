(() => {
  const state = { admin: Boolean(window.COVALT && window.COVALT.admin), polling: null };
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
  let toastTimer;
  const toast = (message) => {
    toastNode.textContent = message;
    toastNode.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toastNode.classList.remove('show'), 3600);
  };
  const api = async (url, options = {}) => {
    const response = await fetch(url, { headers: { 'Content-Type': 'application/json', ...(options.headers || {}) }, ...options });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.error || 'Что-то пошло не так');
    return data;
  };

  function renderVideos(videos) {
    $('#libraryCount').textContent = `${videos.length} ${videos.length === 1 ? 'работа' : 'работ'}`;
    if (!videos.length) {
      videoGrid.innerHTML = '<div class="empty-state"><span class="empty-ring"></span><p>Здесь появятся опубликованные сцены.</p><small>Сгенерируйте первое видео выше.</small></div>';
      return;
    }
    videoGrid.innerHTML = videos.map((video) => {
      const draft = !video.published;
      const actions = draft && state.admin
        ? `<button class="publish-button" data-action="publish" data-id="${escapeHtml(video.id)}">Опубликовать ↗</button><button class="delete-button" data-action="delete" data-id="${escapeHtml(video.id)}">Удалить</button>`
        : `<a class="download-button" href="${escapeHtml(video.download_url)}">Скачать MP4 ↓</a>`;
      return `<article class="video-card">
        <div class="video-frame"><video controls preload="metadata" playsinline poster="${escapeHtml(video.thumbnail_url || '')}" src="${escapeHtml(video.video_url || '')}"></video></div>
        <div class="video-info"><div class="video-top"><h3 class="video-title" title="${escapeHtml(video.title)}">${escapeHtml(video.title)}</h3><span class="video-duration">${formatDuration(video.duration)}</span></div>
        <p class="video-prompt">${escapeHtml(video.prompt)}</p><div class="video-footer"><span class="video-date">${draft ? '<span class="draft-badge">DRAFT</span>' : formatDate(video.created_at)}</span><span>${actions}</span></div></div>
      </article>`;
    }).join('');
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
    if (state.admin) { document.querySelector('#library').scrollIntoView({ behavior: 'smooth' }); return; }
    modal.hidden = false; setTimeout(() => $('#password').focus(), 20);
  });
  $('#closeModal').addEventListener('click', () => { modal.hidden = true; });
  modal.addEventListener('click', (event) => { if (event.target === modal) modal.hidden = true; });
  document.addEventListener('keydown', (event) => { if (event.key === 'Escape') modal.hidden = true; });
  $('#loginForm').addEventListener('submit', async (event) => {
    event.preventDefault();
    $('#loginError').textContent = '';
    try {
      await api('/api/login', { method: 'POST', body: JSON.stringify({ password: $('#password').value }) });
      state.admin = true; modal.hidden = true; $('#password').value = ''; updateAdminUi(); await refreshVideos(); toast('Режим администратора включён.');
    } catch (error) { $('#loginError').textContent = error.message; }
  });
  $('#logoutButton').addEventListener('click', async () => { try { await api('/api/logout', { method: 'POST' }); state.admin = false; updateAdminUi(); await refreshVideos(); toast('Вы вышли из режима администратора.'); } catch (error) { toast(error.message); } });

  videoGrid.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-action]');
    if (!button) return;
    const id = button.dataset.id;
    button.disabled = true;
    try {
      if (button.dataset.action === 'publish') { await api(`/api/videos/${encodeURIComponent(id)}/publish`, { method: 'POST' }); toast('Видео опубликовано в библиотеке.'); }
      if (button.dataset.action === 'delete') { if (!window.confirm('Удалить это видео навсегда?')) { button.disabled = false; return; } await api(`/api/videos/${encodeURIComponent(id)}`, { method: 'DELETE' }); toast('Видео удалено.'); }
      await refreshVideos();
    } catch (error) { button.disabled = false; toast(error.message); }
  });

  updateAdminUi();
  refreshVideos();
})();
