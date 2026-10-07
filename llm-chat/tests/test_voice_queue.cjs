// Run: node tests/test_voice_queue.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const path = require("node:path");
const html = fs.readFileSync(path.join(__dirname, "../src/llm_chat/static/index.html"), "utf8");
const functions = html.slice(html.indexOf("function createSpeech("), html.indexOf("// ブラウザーで録音"));

async function run(engine, id, style) {
  const payloads = [], events = [], revoked = [];
  const context = vm.createContext({
    state: {}, AbortController,
    $: () => ({ textContent: "" }),
    speechText: text => text.trim(),
    URL: { createObjectURL: () => `blob:${payloads.length}`, revokeObjectURL: url => revoked.push(url) },
    fetch: async (url, options) => {
      payloads.push(JSON.parse(options.body)); events.push("fetch");
      return { ok: true, blob: async () => ({}) };
    },
    Audio: class {
      play() { events.push("play"); return Promise.resolve(); }
      pause() { events.push("pause"); }
    },
  });
  vm.runInContext(functions, context);
  const speech = context.createSpeech({ engine, id }, style);
  context.state.speech = speech;
  for (let index = 0; index < 6; index++) context.queueSpeech(speech, `文${index}です。`);
  for (let index = 0; index < 50; index++) await Promise.resolve();
  assert.deepEqual(events.slice(0, 3), ["fetch", "play", "fetch"]);
  assert.equal(payloads.length, 3, "one playing clip plus two completed clips limits prefetch");
  if (engine === "irodori") {
    assert.equal(payloads[0].preset, "happy");
    assert.equal(payloads[0].voice_id, id);
    assert.equal(payloads[0].style_id, undefined);
  } else {
    assert.equal(payloads[0].style_id, 3);
    assert.equal(payloads[0].preset, undefined);
  }
  context.stopSpeech();
  for (let index = 0; index < 50; index++) await Promise.resolve();
  assert.equal(speech.stopped, true);
  assert.equal(speech.abort.signal.aborted, true);
  assert.equal(speech.queue.length, 0);
  assert.equal(speech.ready.length, 0);
  assert.equal(revoked.length, 3, "stop releases playing and prefetched object URLs");
  assert.equal(payloads.length, 3, "stop never sends a runtime OFF request");
}

(async () => {
  await run("irodori", "voice-a", "happy");
  await run("voicevox", null, "3");
  console.log("PASS: caption presets, legacy IDs, prefetch during playback, bounded buffering and stop");
})().catch(error => { console.error(error); process.exitCode = 1; });
