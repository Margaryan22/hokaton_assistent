// Голосовой слой браузера.
//
// Основной путь — модели на сервере: микрофон записывается здесь (AudioWorklet → 16 кГц → WAV),
// распознаёт GigaAM v3 (POST /api/stt), озвучивает Silero (POST /api/tts).
// Запасной путь, если серверные модели выключены, — встроенные Web Speech API и speechSynthesis.

const TARGET_RATE = 16000;
const MAX_SECONDS = 20;          // GigaAM рассчитан на фразы до ~25 секунд
const SILENCE_TO_STOP = 1.1;     // секунд тишины после речи — конец фразы
const WAIT_FOR_SPEECH = 7;       // если за это время не заговорили — останавливаемся

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
    this.active = false;
    this.supported = canRecord;
  }

  async start() {
    if (this.active) return;
    stopSpeaking();
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (error) {
      this.handlers.onError?.(error.name === "NotAllowedError" ? "not-allowed" : error.message);
      return;
    }
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
    this.#setActive(true);
  }

  #onAudio(chunk) {
    if (!this.active) return;
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
    if (this.heardSpeechAt && now - this.lastLoudAt > SILENCE_TO_STOP) this.stop();
    else if (!this.heardSpeechAt && elapsed > WAIT_FOR_SPEECH) this.cancel();
    else if (elapsed > MAX_SECONDS) this.stop();
  }

  async stop() {
    if (!this.active) return;
    const audio = downsample(this.chunks, this.context.sampleRate);
    const spoke = Boolean(this.heardSpeechAt);
    this.#release();
    if (!spoke) return;
    this.handlers.onInterim?.("Распознаю…");
    try {
      const response = await fetch("/api/stt", { method: "POST", headers: { "Content-Type": "audio/wav" }, body: encodeWav(audio, TARGET_RATE) });
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const { text } = await response.json();
      this.handlers.onInterim?.("");
      if (text) this.handlers.onFinal?.(text);
      else this.handlers.onError?.("no-speech");
    } catch (error) {
      this.handlers.onInterim?.("");
      this.handlers.onError?.("server");
    }
  }

  cancel() {
    this.#release();
  }

  toggle() {
    this.active ? this.stop() : this.start();
  }

  #release() {
    this.node?.port && (this.node.port.onmessage = null);
    this.stream?.getTracks().forEach((t) => t.stop());
    this.context?.close();
    this.#setActive(false);
    this.handlers.onLevel?.(0);
  }

  #setActive(value) {
    this.active = value;
    this.handlers.onStateChange?.(value);
  }
}

// --- запасной вариант: распознавание браузера -----------------------------------

class BrowserListener {
  constructor(handlers) {
    this.handlers = handlers;
    this.active = false;
    this.supported = Boolean(BrowserRecognition);
    if (!this.supported) return;
    this.recognition = new BrowserRecognition();
    this.recognition.lang = "ru-RU";
    this.recognition.interimResults = true;
    this.recognition.continuous = false;
    this.recognition.onresult = (event) => {
      let interim = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i];
        if (result.isFinal) handlers.onFinal?.(result[0].transcript.trim());
        else interim += result[0].transcript;
      }
      if (interim) handlers.onInterim?.(interim.trim());
    };
    this.recognition.onerror = (event) => {
      if (event.error !== "aborted") handlers.onError?.(event.error);
    };
    this.recognition.onend = () => this.#setActive(false);
  }

  start() {
    if (!this.supported || this.active) return;
    stopSpeaking();
    try {
      this.recognition.start();
      this.#setActive(true);
    } catch { /* уже идёт */ }
  }

  stop() { if (this.active) this.recognition.stop(); }
  toggle() { this.active ? this.stop() : this.start(); }

  #setActive(value) {
    this.active = value;
    this.handlers.onStateChange?.(value);
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
