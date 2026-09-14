(() => {
  const $ = (selector) => document.querySelector(selector);
  const list = $('#testList');
  const summary = $('#testSummary');
  const button = $('#runTests');
  const escapeHtml = (value) => String(value ?? '').replace(/[&<>'"]/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[char]));
  const render = (data) => {
    const allPassed = data.passed === data.total;
    summary.className = `test-summary ${allPassed ? 'ok' : 'attention'}`;
    summary.innerHTML = `<b>${data.passed}/${data.total}</b> проверок PASS <span>·</span> запуск ${new Intl.DateTimeFormat('ru-RU', { time: 'medium' }).format(new Date(data.ran_at))}`;
    list.innerHTML = (data.results || []).map((test, index) => `<article class="test-row"><span class="test-number">${String(index + 1).padStart(2, '0')}</span><div><h3>${escapeHtml(test.title)}</h3><p>${escapeHtml(test.description)}</p></div><div class="test-expected"><span>ОЖИДАЛОСЬ</span>${escapeHtml(test.expected)}</div><div class="test-actual"><span>ПОЛУЧЕНО</span>${escapeHtml(test.actual)}${test.details ? `<br>${escapeHtml(test.details).slice(0, 350)}` : ''}</div><span class="test-status ${test.passed ? 'pass' : 'check'}">${test.status}</span></article>`).join('');
    $('#testNote').textContent = data.note || 'Результаты показываются как есть.';
  };
  const run = async () => {
    button.disabled = true; button.querySelector('span').textContent = 'Проверяем…';
    summary.className = 'test-summary'; summary.innerHTML = '<span class="summary-spinner"></span><span>Запускаю проверки Covalt…</span>';
    try {
      const response = await fetch('/api/tests', { cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Тесты недоступны');
      render(data);
    } catch (error) {
      summary.className = 'test-summary attention'; summary.textContent = `Не удалось запустить тесты: ${error.message}`;
    }
    button.disabled = false; button.querySelector('span').textContent = 'Запустить тесты';
  };
  button.addEventListener('click', run);
  run();
})();
