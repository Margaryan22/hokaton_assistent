// Чистые функции ввода: node --test tests/js
import assert from "node:assert/strict";
import { test } from "node:test";

import { actionFor, appendToInput, formatTimer } from "../../static/input.js";

test("appendToInput дописывает через один пробел", () => {
  assert.equal(appendToInput("", "завтра в десять"), "завтра в десять");
  assert.equal(appendToInput("завтра в десять позвонить Ивану и Петру", "срочно"),
    "завтра в десять позвонить Ивану и Петру срочно");
  assert.equal(appendToInput("купить хлеб  ", "  и молоко "), "купить хлеб и молоко");
  assert.equal(appendToInput("текст", "много   пробелов"), "текст много пробелов");
});

test("appendToInput не ставит пробел перед знаками препинания", () => {
  assert.equal(appendToInput("позвонить Ивану", ", срочно"), "позвонить Ивану, срочно");
  assert.equal(appendToInput("Что у меня завтра", "?"), "Что у меня завтра?");
  assert.equal(appendToInput("", "привет , как дела ?"), "привет, как дела?");
});

test("appendToInput без нового текста оставляет поле как было", () => {
  assert.equal(appendToInput("как было ", ""), "как было ");
  assert.equal(appendToInput("как было", "   "), "как было");
});

test("Esc и Enter работают только во время записи", () => {
  assert.equal(actionFor("recording", "escape"), "cancel");
  assert.equal(actionFor("recording", "enter"), "finish");
  assert.equal(actionFor("idle", "enter"), null);
  assert.equal(actionFor("idle", "escape"), null);
  assert.equal(actionFor("transcribing", "enter"), null);
});

test("кнопки панели записи", () => {
  assert.equal(actionFor("idle", "mic"), "start");
  assert.equal(actionFor("recording", "cancel"), "cancel");
  assert.equal(actionFor("recording", "done"), "finish");
  assert.equal(actionFor("recording", "sendNow"), "finishAndSend");
  // Во время распознавания работает только ✕.
  assert.equal(actionFor("transcribing", "cancel"), "cancel");
  assert.equal(actionFor("transcribing", "done"), null);
  assert.equal(actionFor("transcribing", "sendNow"), null);
  assert.equal(actionFor("transcribing", "mic"), null);
});

test("formatTimer", () => {
  assert.equal(formatTimer(0), "0:00");
  assert.equal(formatTimer(7.9), "0:07");
  assert.equal(formatTimer(75), "1:15");
});
