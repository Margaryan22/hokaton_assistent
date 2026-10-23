// Строка ввода: чистые функции без DOM (их проверяет node --test tests/js/*.test.mjs).

// Что делает кнопка или клавиша в каждом состоянии ввода. Состояния: idle → recording → transcribing → idle.
const ACTIONS = {
  idle: { mic: "start", space: "start" },
  recording: {
    cancel: "cancel", escape: "cancel", done: "finish", enter: "finish", space: "finish", sendNow: "finishAndSend",
  },
  // Пока идёт распознавание, можно только отменить ожидание.
  transcribing: { cancel: "cancel", escape: "cancel" },
};

export function actionFor(state, event) {
  return ACTIONS[state]?.[event] ?? null;
}

// Дописывает распознанный текст в конец поля: один пробел между частями, без пробела перед знаками препинания.
export function appendToInput(current, addition) {
  const tidy = (s) => s.replace(/\s+/g, " ").replace(/ ([.,!?;:…»)])/g, "$1");
  const add = tidy(addition || "").trim();
  if (!add) return current;
  const base = (current || "").trimEnd();
  if (!base) return add;
  return /^[.,!?;:…»)]/.test(add) ? base + add : `${base} ${add}`;
}

export function formatTimer(seconds) {
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
