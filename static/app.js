import { actionFor, appendToInput, formatTimer } from "/static/input.js";
import { createListener, MAX_SECONDS, speak } from "/static/voice.js";

const $ = (id) => document.getElementById(id);
const els = {
  messages: $("messages"), text: $("text"), composer: $("composer"), mic: $("mic"), hint: $("hint"),
  status: $("status"), user: $("user"), reset: $("reset"), tts: $("tts-toggle"), agenda: $("agenda"),
  ics: $("ics"), toasts: $("toasts"), suggestions: $("suggestions"), models: $("models"),
  recorder: $("recorder"), recCancel: $("rec-cancel"), recDone: $("rec-done"), recSend: $("rec-send"),
  recCaption: $("rec-caption-text"), recWave: $("rec-wave"), recTimer: $("rec-timer"), recStatus: $("rec-status"),
};
const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;
const store = {
  get(key, fallback) { try { return localStorage.getItem(key) ?? fallback; } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, value); } catch { /* приватный режим */ } },
};

// Какие голосовые модели есть на сервере, узнаём из /api/health.
const voice = { serverStt: false, serverTts: false };
let listener = null;
let ttsEnabled = store.get("tts", "1") === "1";
let lastInputWasVoice = false;
let agendaRange = 1;
let lastReminderCheck = new Date();

// --- пользователь и сессия ---------------------------------------------------

function userId() {
  return (els.user.value.trim() || "demo").replace(/[^\w.-]/g, "").slice(0, 50) || "demo";
}

function sessionId() {
  const key = `session:${userId()}`;
  let id = store.get(key, null);
  if (!id) {
    id = (crypto.randomUUID?.() || String(Date.now()) + Math.random().toString(16).slice(2)).replace(/[^\w-]/g, "");
    store.set(key, id);
  }
  return id;
}

// --- отрисовка ---------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function card(item) {
  const node = el("div", `card${item.status === "done" ? " done" : ""}`);
  node.append(el("span", `badge ${item.type}`, item.type_label));
  node.append(el("span", "title", item.title));
  node.append(el("span", "when", item.when));
  return node;
}

function addMessage(role, text, { items = [], numbered = false, meta = "", awaiting = false } = {}) {
  const msg = el("div", `msg ${role}${awaiting ? " awaiting" : ""}`);
  msg.append(el("div", "bubble", text));
  if (items.length) {
    const cards = el("div", "cards");
    items.slice(0, 15).forEach((item, i) => {
      const c = card(item);
      if (numbered) c.querySelector(".title").textContent = `${i + 1}. ${item.title}`;
      cards.append(c);
    });
    if (items.length > 15) cards.append(el("div", "meta", `…и ещё ${items.length - 15} — уточните запрос`));
    msg.append(cards);
  }
  if (meta) msg.append(el("div", "meta", meta));
  els.messages.append(msg);
  els.messages.scrollTop = els.messages.scrollHeight;
  return msg;
}

function typing() {
  const msg = el("div", "msg assistant");
  const bubble = el("div", "bubble");
  const dots = el("span", "typing");
  dots.append(el("span"), el("span"), el("span"));
  bubble.append(dots);
  msg.append(bubble);
  els.messages.append(msg);
  els.messages.scrollTop = els.messages.scrollHeight;
  return msg;
}

// --- запросы -----------------------------------------------------------------

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  return response.status === 204 ? null : response.json();
}

async function send(text, viaVoice = false) {
  text = text.trim();
  if (!text) return;
  lastInputWasVoice = viaVoice;
  els.text.value = "";
  els.suggestions.hidden = true;
  addMessage("user", text, { meta: viaVoice ? "голосом" : "" });
  const pending = typing();
  try {
    const reply = await api("/api/chat", {
      method: "POST",
      body: JSON.stringify({ text, session_id: sessionId(), user_id: userId(), timezone }),
    });
    pending.remove();
    const meta = { local: "разобрано локальной LLM", claude: "разобрано Claude", rules: "разобрано правилами" }[reply.nlu] || "";
    addMessage("assistant", reply.summary, { items: reply.items, numbered: reply.numbered, meta, awaiting: Boolean(reply.awaiting) });
    respondAloud(reply);
    loadAgenda();
  } catch (error) {
    pending.remove();
    addMessage("assistant", "Не удалось связаться с сервером. Попробуйте ещё раз.");
    console.error(error);
  }
}

function respondAloud(reply) {
  // Если ассистент ждёт ответа, а пользователь говорил голосом, — сразу слушаем дальше.
  const continueListening = () => {
    if (reply.awaiting && lastInputWasVoice) startRecording("handsfree");
  };
  if (ttsEnabled) speak(reply.speech, { server: voice.serverTts, onEnd: continueListening });
  else continueListening();
}

// --- повестка ----------------------------------------------------------------

const MONTHS = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа", "сентября", "октября", "ноября", "декабря"];
const WEEKDAYS = ["воскресенье", "понедельник", "вторник", "среда", "четверг", "пятница", "суббота"];

function dayTitle(iso) {
  const [y, m, d] = iso.split("-").map(Number);
  const date = new Date(y, m - 1, d);
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const diff = Math.round((date - today) / 86400000);
  const rel = { 0: "Сегодня", 1: "Завтра", 2: "Послезавтра" }[diff];
  return `${rel || WEEKDAYS[date.getDay()]}, ${d} ${MONTHS[m - 1]}`;
}

async function loadAgenda() {
  const start = new Date();
  const end = new Date(); end.setDate(end.getDate() + agendaRange - 1);
  const fmt = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
  try {
    const data = await api(`/api/agenda?user_id=${encodeURIComponent(userId())}&start=${fmt(start)}&end=${fmt(end)}&tz=${encodeURIComponent(timezone)}`);
    renderAgenda(data.items);
  } catch (error) {
    els.agenda.replaceChildren(el("div", "empty", "Не удалось загрузить повестку"));
  }
}

function renderAgenda(items) {
  closeConfirm?.();
  const groups = new Map();
  for (const item of items) {
    const key = item.day || "none";
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  const nodes = [];
  for (const [key, list] of groups) {
    const day = el("div", "day");
    day.append(el("h3", "", key === "none" ? "Без срока" : dayTitle(key)));
    const cards = el("div", "cards");
    list.forEach((item) => {
      const c = card(item);
      c.querySelector(".when").textContent = key === "none" ? "без срока" : item.time ? `${item.time}${item.recurrence ? " · повторяется" : ""}` : item.is_deadline ? "срок сегодня" : "весь день";
      const actions = recordActions(item, c);
      c.classList.add("with-actions");
      c.style.setProperty("--actions", String(actions.children.length));
      c.append(actions);
      cards.append(c);
    });
    day.append(cards);
    nodes.push(day);
  }
  if (!nodes.length) nodes.push(el("div", "empty", agendaRange === 1 ? "На сегодня ничего не запланировано." : "В этом периоде записей нет."));
  els.agenda.replaceChildren(...nodes);
}

// Контуры иконок — те же Material-образцы, что и у кнопок записи в index.html.
const ICON_PATHS = {
  complete: "M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4z",
  delete: "M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z",
};

function button(label, className, onClick) {
  const node = el("button", className, label);
  node.type = "button";
  node.addEventListener("click", onClick);
  return node;
}

function iconButton(icon, label, className, onClick) {
  const node = button("", className, onClick);
  node.setAttribute("aria-label", label);
  node.title = label;
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("viewBox", "0 0 24 24");
  svg.setAttribute("aria-hidden", "true");
  const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
  path.setAttribute("d", ICON_PATHS[icon]);
  svg.append(path);
  node.append(svg);
  return node;
}

// Кнопки записи в повестке — иконки в углу карточки. Удаление подтверждается слоем поверх карточки.
function recordActions(item, cardNode) {
  const box = el("div", "card-actions");
  if (item.type !== "meeting" && item.status !== "done") {
    box.append(iconButton("complete", "Отметить выполненным", "", () => recordAction(item, "complete")));
  }
  const remove = iconButton("delete", "Удалить", "danger", () => confirmDelete(item, cardNode, remove));
  box.append(remove);
  return box;
}

// Закрывает открытое подтверждение удаления; одновременно открыто не больше одного.
let closeConfirm = null;

function confirmDelete(item, cardNode, opener) {
  closeConfirm?.();
  const question = item.recurrence ? "Удалить всю серию?" : `Удалить «${item.title}»?`;
  const text = el("span", "confirm-text", question);
  text.title = question;
  const yes = button("Да", "danger", () => { detach(); yes.disabled = no.disabled = true; recordAction(item, "delete"); });
  const no = button("Нет", "", () => close({ refocus: true }));
  const buttons = el("div", "confirm-buttons");
  buttons.append(yes, no);
  const layer = el("div", "card-confirm");
  layer.setAttribute("role", "group");
  layer.setAttribute("aria-label", question);
  layer.append(text, buttons);

  const onKey = (event) => { if (event.key === "Escape") close({ refocus: true }); };
  const onOutside = (event) => { if (!layer.contains(event.target)) close(); };
  const detach = () => {
    document.removeEventListener("keydown", onKey);
    document.removeEventListener("pointerdown", onOutside, true);
    closeConfirm = null;
  };
  const close = ({ refocus = false } = {}) => {
    detach();
    layer.remove();
    if (refocus) opener.focus();
  };
  document.addEventListener("keydown", onKey);
  document.addEventListener("pointerdown", onOutside, true);
  closeConfirm = close;
  cardNode.append(layer);
  no.focus();
}

async function recordAction(item, action) {
  const path = `/api/records/${item.id}${action === "complete" ? "/complete" : ""}?user_id=${encodeURIComponent(userId())}&tz=${encodeURIComponent(timezone)}`;
  try {
    await api(path, { method: action === "complete" ? "POST" : "DELETE" });
    toast(action === "complete" ? "Выполнено:" : "Удалено:", item.title, [], { kind: "info", timeout: 4000 });
  } catch (error) {
    toast("Не получилось", "Запись могла быть уже удалена. Повестка обновлена.", [], { kind: "info", timeout: 6000 });
    console.error(error);
  }
  loadAgenda();
  loadStatus();
}

// --- напоминания -------------------------------------------------------------

function toast(title, body, actions = [], { kind = "", timeout = 0 } = {}) {
  const node = el("div", `toast${actions.length ? " error" : ""}${kind ? ` ${kind}` : ""}`);
  node.append(el("strong", "", title), el("span", "", body));
  if (actions.length) {
    const row = el("div", "toast-actions");
    for (const { label, run } of actions) {
      const button = el("button", "", label);
      button.type = "button";
      button.addEventListener("click", () => { node.remove(); run(); });
      row.append(button);
    }
    node.append(row);
  }
  els.toasts.append(node);
  setTimeout(() => node.remove(), timeout || (actions.length ? 60000 : 15000));
  return node;
}

async function checkReminders() {
  const since = lastReminderCheck;
  lastReminderCheck = new Date();
  const local = new Date(since.getTime() - since.getTimezoneOffset() * 60000).toISOString().slice(0, 19);
  try {
    const due = await api(`/api/reminders/due?user_id=${encodeURIComponent(userId())}&since=${local}&tz=${encodeURIComponent(timezone)}`);
    for (const item of due) {
      toast("Напоминание", item.title);
      if (ttsEnabled) speak(`Напоминание: ${item.title}`, { server: voice.serverTts });
      if ("Notification" in window && Notification.permission === "granted") new Notification("Напоминание", { body: item.title });
    }
  } catch { /* сеть недоступна — попробуем в следующий раз */ }
}

// --- инициализация -----------------------------------------------------------

// --- голосовой ввод ------------------------------------------------------------
//
// Диктовка (🎤): запись идёт, пока пользователь не нажмёт ✓ (текст вставится в поле), ➤ (вставится и уйдёт
// в чат) или ✕ (запись выброшена). Hands-free — после голосового вопроса ассистента: запись останавливается
// на паузе, и ответ уходит сам. Состояния ввода приходят от слушателя: idle → recording → transcribing → idle.

const BAR_COUNT = 120;  // с запасом на широкий экран: лишние полоски обрезаются слева
const BAR_STEP_MS = 70;
const ERROR_MESSAGES = {
  "not-allowed": "Нет доступа к микрофону — разрешите его в настройках браузера.",
  "no-speech": "Ничего не расслышал. Попробуйте ещё раз.",
  network: "Распознавание речи недоступно: нет сети.",
  server: "Сервер не смог распознать запись. Попробуйте ещё раз или напишите текстом.",
};
const rec = { state: "idle", mode: "dictation", startedAt: 0, frame: 0, peak: 0, levels: [], lastBar: 0, warned: false, interim: "" };

function initWave() {
  els.recWave.replaceChildren(...Array.from({ length: BAR_COUNT }, (_, i) => {
    const bar = document.createElement("i");
    bar.style.setProperty("--i", String(BAR_COUNT - i));
    return bar;
  }));
}

function announce(text) {
  els.recStatus.textContent = text;
}

// Промежуточный текст Web Speech — одной строкой, обрезанной слева; иначе короткая подпись режима.
function setCaption() {
  els.recCaption.parentElement.classList.toggle("rec-caption--interim", Boolean(rec.interim));
  els.recCaption.textContent = rec.interim
    || (rec.mode === "handsfree" ? "Слушаю ответ… замолчите, чтобы отправить" : "Говорите…");
}

function focusInputEnd() {
  els.text.focus();
  const end = els.text.value.length;
  els.text.setSelectionRange(end, end);
}

function renderRecorder(state) {
  const previous = rec.state;
  rec.state = state;
  els.recorder.hidden = state === "idle";
  els.composer.classList.toggle("composer--recording", state === "recording");
  els.composer.classList.toggle("composer--transcribing", state === "transcribing");
  els.recorder.classList.toggle("recorder--breathing", state === "recording" && !listener.hasLevel);
  els.recDone.disabled = els.recSend.disabled = state !== "recording";
  // В hands-free ✓ и так отправляет сразу — отдельная ➤ не нужна.
  els.recSend.hidden = rec.mode === "handsfree";
  els.recDone.setAttribute("aria-label", rec.mode === "handsfree" ? "Отправить" : "Готово");
  cancelAnimationFrame(rec.frame);

  if (state === "recording") {
    Object.assign(rec, { startedAt: performance.now(), peak: 0, lastBar: 0, warned: false, interim: "" });
    rec.levels = new Array(BAR_COUNT).fill(0.06);
    for (const bar of els.recWave.children) bar.style.setProperty("--v", "0.06");
    els.recTimer.textContent = formatTimer(0);
    els.hint.textContent = rec.mode === "handsfree"
      ? "✓ — отправить сейчас, ✕ или Esc — отменить."
      : "✓ или Enter — вставить текст в поле, ➤ — отправить сразу, ✕ или Esc — отменить.";
    setCaption();
    announce(rec.mode === "handsfree" ? "Слушаю ответ" : "Запись началась");
    els.recorder.focus();
    rec.frame = requestAnimationFrame(tick);
  } else if (state === "transcribing") {
    els.recCaption.textContent = "Распознаю…";
    announce("Распознаю");
  } else {
    els.recTimer.classList.remove("warn");
    els.hint.textContent = els.hint.dataset.idle || "";
    announce("");
    if (previous !== "idle" && rec.mode === "dictation") focusInputEnd();
  }
}

function tick(now) {
  const elapsed = (now - rec.startedAt) / 1000;
  els.recTimer.textContent = formatTimer(elapsed);
  const warn = elapsed >= MAX_SECONDS - 5;
  els.recTimer.classList.toggle("warn", warn);
  if (warn && !rec.warned) {
    rec.warned = true;
    announce("Осталось 5 секунд записи");
  }
  // Полоски бегут справа налево: новое значение громкости — справа.
  if (listener.hasLevel && now - rec.lastBar >= BAR_STEP_MS) {
    rec.lastBar = now;
    rec.levels.push(Math.max(0.06, rec.peak));
    rec.levels.shift();
    rec.peak = 0;
    const bars = els.recWave.children;
    for (let i = 0; i < bars.length; i++) bars[i].style.setProperty("--v", rec.levels[i].toFixed(2));
  }
  rec.frame = requestAnimationFrame(tick);
}

async function startRecording(mode) {
  if (!listener?.supported || rec.state !== "idle") return;
  rec.mode = mode;
  await listener.start({ autoStop: mode === "handsfree" });
}

// Кнопка или клавиша → действие для текущего состояния ввода (таблица в input.js).
function handleInput(event) {
  const action = actionFor(rec.state, event);
  if (!action || !listener) return false;
  if (action === "start") startRecording("dictation");
  else if (action === "cancel") listener.cancel();
  else if (action === "finish") complete(listener.finish(), { sendNow: rec.mode === "handsfree" });
  else if (action === "finishAndSend") complete(listener.finish(), { sendNow: true });
  return true;
}

// Распознанный текст дописывается в поле; при sendNow — сразу уходит в чат.
async function complete(result, { sendNow = false, note = "" } = {}) {
  let text;
  try {
    text = await result;
  } catch (error) {
    if (error.code !== "cancelled" && error.code !== "no-audio") showSttError(error.code, sendNow);
    return;
  }
  if (!text) {
    els.hint.textContent = ERROR_MESSAGES["no-speech"];
    return;
  }
  const full = appendToInput(els.text.value, text);
  if (sendNow) {
    send(full, true);
    return;
  }
  els.text.value = full;
  focusInputEnd();
  if (note) els.hint.textContent = note;
}

function showSttError(code, sendNow) {
  if (!listener.canRetry || code === "not-allowed") {
    els.hint.textContent = ERROR_MESSAGES[code] || `Ошибка распознавания: ${code}`;
    return;
  }
  // Запись не теряем: её можно отправить на распознавание ещё раз.
  toast("Не удалось распознать", "Запись сохранена — можно попробовать ещё раз.", [
    { label: "Повторить", run: () => { if (rec.state === "idle") complete(listener.retry(), { sendNow }); } },
    { label: "Отмена", run: () => listener.discard() },
  ]);
}

function onAutoStop(reason, result) {
  if (reason === "limit") complete(result, { sendNow: rec.mode === "handsfree", note: "Достигнут лимит записи." });
  else if (reason === "silence") complete(result, { sendNow: true });
}

function buildListener() {
  listener = createListener({
    server: voice.serverStt,
    onInterim: (text) => {
      rec.interim = text;
      if (rec.state === "recording") setCaption();
    },
    onLevel: (level) => { rec.peak = Math.max(rec.peak, level); },
    onStateChange: renderRecorder,
    onAutoStop,
    onError: (error) => { els.hint.textContent = ERROR_MESSAGES[error] || `Ошибка распознавания: ${error}`; },
  });
  els.mic.disabled = !listener.supported;
  els.hint.dataset.idle = !listener.supported
    ? "Голосовой ввод в этом браузере недоступен — можно писать текстом."
    : "Нажмите на микрофон (или пробел) и говорите: текст появится в поле, его можно поправить перед отправкой.";
  if (!listener.active) els.hint.textContent = els.hint.dataset.idle;
}

function setTts(value) {
  ttsEnabled = value && (voice.serverTts || "speechSynthesis" in window);
  els.tts.setAttribute("aria-pressed", String(ttsEnabled));
  els.tts.textContent = ttsEnabled ? "🔊 Озвучка" : "🔇 Озвучка";
  store.set("tts", ttsEnabled ? "1" : "0");
}

async function loadStatus() {
  try {
    const health = await api(`/api/health?user_id=${encodeURIComponent(userId())}`);
    voice.serverStt = health.server_stt;
    voice.serverTts = health.server_tts;
    const m = health.models;
    els.status.textContent = `записей: ${health.records}`;
    els.status.title = `Распознавание: ${m.stt}\nСинтез: ${m.tts}\nПонимание: ${m.nlu}\nПоиск: ${m.search}`;
    els.models.textContent = `🎙 ${m.stt} · 🔊 ${m.tts} · 🧠 ${m.nlu} · 🔎 ${m.search}`;
  } catch {
    els.status.textContent = "Сервер недоступен";
  }
  if (!listener?.active) buildListener();
}

function greet() {
  els.messages.replaceChildren();
  els.suggestions.hidden = false;
  addMessage("assistant", "Здравствуйте! Я веду ваши задачи, напоминания и встречи. Скажите, например: «Что у меня сегодня?» или «Напомни завтра в 10 позвонить Ивану».");
}

els.composer.addEventListener("submit", (event) => { event.preventDefault(); send(els.text.value); });
els.mic.addEventListener("click", () => {
  if ("Notification" in window && Notification.permission === "default") Notification.requestPermission();
  handleInput("mic");
});
els.recCancel.addEventListener("click", () => handleInput("cancel"));
els.recDone.addEventListener("click", () => handleInput("done"));
els.recSend.addEventListener("click", () => handleInput("sendNow"));
els.suggestions.addEventListener("click", (event) => {
  if (event.target.tagName === "BUTTON") send(event.target.textContent);
});
els.tts.addEventListener("click", () => setTts(!ttsEnabled));
els.reset.addEventListener("click", () => {
  store.set(`session:${userId()}`, "");
  greet();
});
els.user.value = store.get("user", "demo");
els.user.addEventListener("change", () => {
  store.set("user", userId());
  els.ics.href = `/api/calendar.ics?user_id=${encodeURIComponent(userId())}`;
  greet(); loadAgenda(); loadStatus();
});
document.querySelectorAll(".tabs button").forEach((tab) => tab.addEventListener("click", () => {
  document.querySelectorAll(".tabs button").forEach((t) => t.setAttribute("aria-selected", String(t === tab)));
  agendaRange = Number(tab.dataset.range);
  loadAgenda();
}));
document.addEventListener("keydown", (event) => {
  if (event.isComposing) return;
  // На кнопке панели записи или повестки Enter и пробел нажимают саму кнопку (например, ✕ или «Да» в подтверждении).
  const onRecorderButton = Boolean(event.target.closest?.(".rec-btn, .card-actions, .card-confirm"));
  if (event.key === "Escape") {
    if (handleInput("escape")) event.preventDefault();
  } else if (event.key === "Enter" && !onRecorderButton) {
    // Во время записи Enter — «Готово»; в обычном поле Enter по-прежнему отправляет форму.
    if (handleInput("enter")) event.preventDefault();
  } else if (event.code === "Space" && !onRecorderButton
             && document.activeElement !== els.text && document.activeElement !== els.user) {
    // Пробел вне поля ввода — начать диктовку или закончить её.
    if (handleInput("space")) event.preventDefault();
  }
});

initWave();
els.ics.href = `/api/calendar.ics?user_id=${encodeURIComponent(userId())}`;
greet();
loadStatus().then(() => setTts(ttsEnabled));
loadAgenda();
setInterval(checkReminders, 30000);
