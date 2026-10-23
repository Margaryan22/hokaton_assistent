// Голосовой слой браузера.
//
// Основной путь — модели на сервере: микрофон записывается здесь (AudioWorklet → 16 кГц → WAV),
// распознаёт GigaAM v3 (POST /api/stt), озвучивает Silero (POST /api/tts).
// Запасной путь, если серверные модели выключены, — встроенные Web Speech API и speechSynthesis.
//
// Оба слушателя устроены одинаково:
//   start({ autoStop })  — начать запись; autoStop: true — остановиться самому после паузы (hands-free),
//                          false — писать, пока пользователь не нажмёт «Готово» или «Отменить» (диктовка);
//   finish()             — остановить запись и вернуть Promise с распознанным текстом;
//   cancel()             — выбросить запись или перестать ждать распознавания;
//   retry()              — повторно распознать последнюю запись (после ошибки сети или сервера);
//   события onLevel(0..1), onInterim(text), onStateChange("idle" | "recording" | "transcribing"),
//   onAutoStop(reason, promise) — запись остановилась сама: "silence", "limit" или "no-speech", onError(code).
// Promise из finish()/retry() отклоняется с VoiceError: code "cancelled", "server", "network", "not-allowed"…

const TARGET_RATE = 16000;
export const MAX_SECONDS = 20;   // GigaAM рассчитан на фразы до ~25 секунд
const SILENCE_TO_STOP = 1.1;     // секунд тишины после речи — конец фразы
const WAIT_FOR_SPEECH = 7;       // если за это время не заговорили — останавливаемся

export class VoiceError extends Error {
  constructor(code) {
    super(code);
    this.code = code;
  }
}

// Promise, который не роняет «unhandled rejection», если его никто не ждёт (например, запись отменили).
function quiet(promise) {
  promise.catch(() => {});
  return promise;
}

const BrowserRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const canRecord = Boolean(navigator.mediaDevices?.getUserMedia && window.AudioWorkletNode);

const CAPTURE_WORKLET = `
class Capture extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel) this.port.postMessage(channel.slice(0));
    return true;
  }
}
registerProcessor("capture", Capture);
`;

// --- запись и распознавание на сервере ----------------------------------------

function downsample(chunks, fromRate) {
  const length = chunks.reduce((n, c) => n + c.length, 0);
  const input = new Float32Array(length);
  let offset = 0;
  for (const c of chunks) { input.set(c, offset); offset += c.length; }
  if (fromRate === TARGET_RATE) return input;
  const ratio = fromRate / TARGET_RATE;
  const output = new Float32Array(Math.floor(length / ratio));
  for (let i = 0; i < output.length; i++) {
    // Усредняем отсчёты, попавшие в интервал, — простой фильтр от наложения частот.
    const start = Math.floor(i * ratio), end = Math.min(Math.floor((i + 1) * ratio), length);
    let sum = 0;
    for (let j = start; j < end; j++) sum += input[j];
    output[i] = sum / Math.max(end - start, 1);
  }
  return output;
}

function encodeWav(samples, rate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, text) => [...text].forEach((ch, i) => view.setUint8(offset + i, ch.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true); write(8, "WAVE");
  write(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, rate, true); view.setUint32(28, rate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((s, i) => view.setInt16(44 + i * 2, Math.max(-1, Math.min(1, s)) * 0x7fff, true));
  return new Blob([buffer], { type: "audio/wav" });
}

class ServerListener {
  constructor(handlers) {
    this.handlers = handlers;
    this.state = "idle";
    this.supported = canRecord;
    this.hasLevel = true;    // есть настоящий уровень громкости для визуализации
    this.canRetry = true;    // запись хранится, её можно отправить на распознавание ещё раз
    this.lastWav = null;
  }

  get active() { return this.state !== "idle"; }

  // Возвращает true, если запись началась.
  async start({ autoStop = true } = {}) {
    if (this.state !== "idle" || this.session) return false;
    stopSpeaking();
    const session = this.session = {};
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (error) {
      if (this.session === session) this.session = null;
      this.handlers.onError?.(error.name === "NotAllowedError" ? "not-allowed" : error.message);
      return false;
    }
    if (this.session !== session) {  // пока ждали разрешения, запись отменили
      stream.getTracks().forEach((t) => t.stop());
      return false;
    }
    this.stream = stream;
    this.autoStop = autoStop;
    this.context = new AudioContext();
    const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" }));
    await this.context.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const source = this.context.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.context, "capture");
    this.chunks = [];
    this.startedAt = this.context.currentTime;
    this.heardSpeechAt = null;
    this.lastLoudAt = null;
    this.noise = 0.01;
    this.node.port.onmessage = (event) => this.#onAudio(event.data);
    source.connect(this.node);
    this.#setState("recording");
    return true;
  }

  #onAudio(chunk) {
    if (this.state !== "recording") return;
    this.chunks.push(chunk);
    const now = this.context.currentTime, elapsed = now - this.startedAt;
    let energy = 0;
    for (const s of chunk) energy += s * s;
    const rms = Math.sqrt(energy / chunk.length);
    // Простейший детектор речи: громкость заметно выше фонового шума.
    if (elapsed < 0.3) this.noise = Math.max(this.noise, rms);
    const loud = rms > Math.max(0.02, this.noise * 2.5);
    if (loud) {
      this.lastLoudAt = now;
      this.heardSpeechAt ??= now;
    }
    this.handlers.onLevel?.(Math.min(rms * 8, 1));
    if (elapsed > MAX_SECONDS) this.#autoFinish("limit");
    else if (!this.autoStop) return;  // диктовка: останавливает только пользователь
    else if (this.heardSpeechAt && now - this.lastLoudAt > SILENCE_TO_STOP) this.#autoFinish("silence");
    else if (!this.heardSpeechAt && elapsed > WAIT_FOR_SPEECH) {
      this.cancel();
      this.handlers.onAutoStop?.("no-speech", Promise.resolve(""));
    }
  }

  #autoFinish(reason) {
    this.handlers.onAutoStop?.(reason, this.finish());
  }

  finish() {
    if (this.state === "transcribing") return this.pending;
    if (this.state !== "recording") return quiet(Promise.reject(new VoiceError("cancelled")));
    const audio = downsample(this.chunks, this.context.sampleRate);
    const spoke = Boolean(this.heardSpeechAt);
    this.#release();
    // В hands-free тишину не отправляем; в диктовке человек сам нажал «Готово» — распознаём всё, что записано.
    if (this.autoStop && !spoke) {
      this.#setState("idle");
      return Promise.resolve("");
    }
    this.lastWav = encodeWav(audio, TARGET_RATE);
    return this.#transcribe(this.lastWav);
  }

  retry() {
    if (this.state !== "idle" || !this.lastWav) return quiet(Promise.reject(new VoiceError("no-audio")));
    return this.#transcribe(this.lastWav);
  }

  // Забыть сохранённую запись (пользователь отказался повторять).
  discard() { this.lastWav = null; }

  #transcribe(wav) {
    const controller = this.abort = new AbortController();
    this.#setState("transcribing");
    this.pending = quiet((async () => {
      try {
        const response = await fetch("/api/stt", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: wav, signal: controller.signal });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const { text } = await response.json();
        this.lastWav = null;
        return (text || "").trim();
      } catch (error) {
        throw new VoiceError(controller.signal.aborted ? "cancelled" : "server");
      } finally {
        if (this.abort === controller) {
          this.abort = null;
          this.#setState("idle");
        }
      }
    })());
    return this.pending;
  }

  cancel() {
    this.session = null;
    if (this.state === "recording") this.#release();
    if (this.abort) {
      this.abort.abort();
      this.abort = null;
    }
    this.#setState("idle");
  }

  #release() {
    this.session = null;
    this.node?.port && (this.node.port.onmessage = null);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.context?.close();
    this.handlers.onLevel?.(0);
  }

  #setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.handlers.onStateChange?.(state);
  }
}

// --- запасной вариант: распознавание браузера -----------------------------------

class BrowserListener {
  constructor(handlers) {
    this.handlers = handlers;
    this.state = "idle";
    this.supported = Boolean(BrowserRecognition);
    this.hasLevel = false;   // Web Speech не отдаёт громкость — панель покажет «дышащие» полоски
    this.canRetry = false;   // звук остаётся внутри браузера, повторить распознавание нельзя
  }

  get active() { return this.state !== "idle" || Boolean(this.recognition); }

  // autoStop: true — браузер сам заканчивает фразу на паузе; false — непрерывная диктовка до «Готово».
  start({ autoStop = true } = {}) {
    if (!this.supported || this.active) return Promise.resolve(false);
    stopSpeaking();
    this.autoStop = autoStop;
    this.finalText = "";
    this.interim = "";
    this.error = null;
    this.cancelled = false;
    this.result = quiet(new Promise((resolve, reject) => { this.resolve = resolve; this.reject = reject; }));
    const recognition = this.recognition = new BrowserRecognition();
    recognition.lang = "ru-RU";
    recognition.interimResults = true;
    recognition.continuous = !autoStop;
    recognition.onaudiostart = () => this.#setState("recording");
    recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) this.finalText = `${this.finalText} ${result[0].transcript}`.trim();
        else interim += result[0].transcript;
      }
      this.interim = interim.trim();
      this.handlers.onInterim?.(`${this.finalText} ${this.interim}`.trim());
    };
    recognition.onerror = (event) => {
      if (event.error !== "aborted" && event.error !== "no-speech") this.error = event.error;
    };
    recognition.onend = () => this.#onEnd(recognition);
    this.limitTimer = setTimeout(() => {
      if (this.state === "recording") this.handlers.onAutoStop?.("limit", this.finish());
    }, MAX_SECONDS * 1000);
    try {
      recognition.start();
    } catch {
      this.recognition = null;
      return Promise.resolve(false);
    }
    return Promise.resolve(true);
  }

  #onEnd(recognition) {
    if (recognition !== this.recognition) return;
    // Chrome сам обрывает долгую непрерывную сессию — в диктовке молча продолжаем запись.
    if (this.state === "recording" && !this.autoStop && !this.error && !this.cancelled) {
      try { recognition.start(); return; } catch { /* не вышло — завершаем как обычно */ }
    }
    clearTimeout(this.limitTimer);
    this.recognition = null;
    const wasRecording = this.state === "recording";
    // Результат ждёт тот, кто нажал «Готово», или получит обработчик автоостановки.
    const awaited = this.state === "transcribing" || (wasRecording && this.autoStop);
    const text = this.finalText || this.interim;
    this.#setState("idle");
    if (this.cancelled) this.reject(new VoiceError("cancelled"));
    else if (this.error && !text) {
      this.reject(new VoiceError(this.error));
      if (!awaited) this.handlers.onError?.(this.error);
    } else this.resolve(text);
    // В hands-free браузер сам закончил фразу на паузе — для приложения это автоостановка.
    if (wasRecording && this.autoStop && !this.cancelled) this.handlers.onAutoStop?.("silence", this.result);
  }

  finish() {
    if (!this.recognition) return quiet(Promise.reject(new VoiceError("cancelled")));
    if (this.state === "recording") {
      this.#setState("transcribing");
      this.recognition.stop();  // финальные результаты придут до onend
    }
    return this.result;
  }

  cancel() {
    if (!this.recognition) return;
    this.cancelled = true;
    this.recognition.abort();
  }

  retry() { return quiet(Promise.reject(new VoiceError("no-audio"))); }
  discard() {}

  #setState(state) {
    if (this.state === state) return;
    this.state = state;
    this.handlers.onStateChange?.(state);
  }
}

export function createListener({ server, ...handlers }) {
  return server && canRecord ? new ServerListener(handlers) : new BrowserListener(handlers);
}

// --- озвучка ---------------------------------------------------------------------

let currentAudio = null;
let russianVoice = null;

if ("speechSynthesis" in window) {
  const pick = () => {
    const voices = window.speechSynthesis.getVoices();
    russianVoice = voices.find((v) => v.lang?.toLowerCase().startsWith("ru")) || null;
  };
  pick();
  window.speechSynthesis.onvoiceschanged = pick;
}

export async function speak(text, { server = false, onEnd } = {}) {
  stopSpeaking();
  if (!text) return onEnd?.();
  if (server) {
    try {
      const response = await fetch("/api/tts", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }),
      });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const url = URL.createObjectURL(await response.blob());
      currentAudio = new Audio(url);
      currentAudio.onended = currentAudio.onerror = () => { URL.revokeObjectURL(url); onEnd?.(); };
      await currentAudio.play();
      return;
    } catch {
      // сервер недоступен — озвучим средствами браузера
    }
  }
  if (!("speechSynthesis" in window)) return onEnd?.();
  const utterance = new SpeechSynthesisUtterance(text.replace(/[«»"]/g, "").replace(/\s—\s/g, ", ").replace(/\n+/g, ". "));
  utterance.lang = "ru-RU";
  if (russianVoice) utterance.voice = russianVoice;
  utterance.onend = utterance.onerror = () => onEnd?.();
  window.speechSynthesis.speak(utterance);
}

export function stopSpeaking() {
  if (currentAudio) { currentAudio.pause(); currentAudio = null; }
  if ("speechSynthesis" in window) window.speechSynthesis.cancel();
}
