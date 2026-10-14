import { createListener, speak } from "/static/voice.js";

const $ = (id) => document.getElementById(id);
const els = {
  messages: $("messages"), text: $("text"), composer: $("composer"), mic: $("mic"), hint: $("hint"),
  status: $("status"), user: $("user"), reset: $("reset"), tts: $("tts-toggle"), agenda: $("agenda"),
  ics: $("ics"), toasts: $("toasts"), suggestions: $("suggestions"), models: $("models"),
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
  return response.json();
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
    if (reply.awaiting && lastInputWasVoice) listener?.start();
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
      cards.append(c);
    });
    day.append(cards);
    nodes.push(day);
  }
  if (!nodes.length) nodes.push(el("div", "empty", agendaRange === 1 ? "На сегодня ничего не запланировано." : "В этом периоде записей нет."));
  els.agenda.replaceChildren(...nodes);
}

// --- напоминания -------------------------------------------------------------

function toast(title, body) {
  const node = el("div", "toast");
  node.append(el("strong", "", title), el("span", "", body));
  els.toasts.append(node);
  setTimeout(() => node.remove(), 15000);
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

function buildListener() {
  listener = createListener({
    server: voice.serverStt,
    onInterim: (text) => { els.text.value = text; },
    onFinal: (text) => send(text, true),
    onLevel: (level) => els.mic.style.setProperty("--level", level.toFixed(2)),
    onStateChange: (active) => {
      els.mic.classList.toggle("listening", active);
      els.mic.setAttribute("aria-label", active ? "Остановить запись" : "Говорить");
      els.hint.textContent = active
        ? (voice.serverStt ? "Слушаю… запись остановится сама, когда вы замолчите" : "Слушаю… говорите")
        : els.hint.dataset.idle || "";
    },
    onError: (error) => {
      const messages = {
        "not-allowed": "Нет доступа к микрофону — разрешите его в настройках браузера.",
        "no-speech": "Не расслышал. Попробуйте ещё раз.",
        network: "Распознавание речи недоступно: нет сети.",
        server: "Сервер не смог распознать запись. Попробуйте ещё раз или напишите текстом.",
      };
      els.hint.textContent = messages[error] || `Ошибка распознавания: ${error}`;
    },
  });
  els.mic.disabled = !listener.supported;
  els.hint.dataset.idle = !listener.supported
    ? "Голосовой ввод в этом браузере недоступен — можно писать текстом."
    : voice.serverStt
      ? "Нажмите на микрофон (или пробел) и говорите — распознаёт GigaAM на сервере."
      : "Нажмите на микрофон и говорите — или напишите запрос.";
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
  listener?.toggle();
});
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
  // Пробел вне поля ввода — начать/остановить запись.
  if (event.code === "Space" && document.activeElement !== els.text && document.activeElement !== els.user) {
    event.preventDefault();
    listener?.toggle();
  }
});

els.ics.href = `/api/calendar.ics?user_id=${encodeURIComponent(userId())}`;
greet();
loadStatus().then(() => setTts(ttsEnabled));
loadAgenda();
setInterval(checkReminders, 30000);
