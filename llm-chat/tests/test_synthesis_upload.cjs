// Run: node tests/test_synthesis_upload.cjs
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const html = fs.readFileSync(path.join(__dirname, '../src/llm_chat/static/voice-synthesize.html'), 'utf8');
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
new vm.Script(script);
const production = html.slice(html.indexOf('<div id="productionPage">'), html.indexOf('<div id="referencePage"'));
assert.ok(production.includes('value="upload"'), 'the production form offers WAV upload');
assert.ok(production.includes('id="synthesisReferenceFile"'), 'the file input is on the production page');

async function scenario(source, file, extra = {}) {
  const elements = new Map();
  const $ = id => {
    if (!elements.has(id)) elements.set(id, {value: '', files: [], hidden: false, required: false, textContent: ''});
    return elements.get(id);
  };
  const defaults = {
    synthesisSource: source, ttsEngine: 'irodori', synthesisText: 'アップロードした声で読みます。',
    synthesisName: '作品', synthesisVoiceCaption: '', synthesisPreset: 'neutral', synthesisSteps: '40',
    textCFG: '3', captionCFG: '3', speakerCFG: '5', synthesisVoice: 'registered-voice',
  };
  for (const [id, value] of Object.entries(defaults)) $(id).value = value;
  const files = Array.isArray(file) ? file : file ? [file] : [];
  $('synthesisReferenceFile').files = files;
  if (extra.text) $('synthesisText').value = extra.text;
  const calls = [], modes = [];
  let pending, outputUpdates = 0;
  const context = vm.createContext({
    $, FormData,
    state: {sourceProduct: null, uploadReferences: files.map((file, index) => ({file, label: extra.labels?.[index] ?? '', reference_text: extra.transcripts?.[index] ?? ''}))},
    document: {querySelectorAll: () => []},
    engineControls: () => {},
    seed: () => 123,
    qwenSettings: () => ({}),
    action: work => { pending = work(); },
    mode: async value => modes.push(value),
    json: (method, body) => ({method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}),
    api: async (url, options) => {
      calls.push({url, options});
      return {references: [{seconds: 120, trimmed_from_seconds: 121}], voice_source: {kind: 'upload'}};
    },
    outputs: async () => outputUpdates++,
    trimNotice: ref => `${ref.trimmed_from_seconds}秒から先頭${ref.seconds}秒を保存しました。`,
  });
  vm.runInContext(script.slice(script.indexOf('function selectedUploads()'), script.indexOf('function nameEditor(')), context);
  vm.runInContext(script.slice(script.indexOf("$('synthesisForm').onsubmit="), script.indexOf('function showPage(')), context);
  vm.runInContext(script.slice(script.indexOf('function updateSource()'), script.indexOf("$('productionTab').onclick=")), context);
  context.updateSource();
  assert.equal($('uploadVoiceFields').hidden, source !== 'upload');
  assert.equal($('synthesisReferenceFile').required, source === 'upload');
  let prevented = false;
  $('synthesisForm').onsubmit({preventDefault() { prevented = true; }});
  assert.equal(prevented, true);
  return {pending, calls, modes, $, get outputUpdates() { return outputUpdates; }};
}

(async () => {
  const file = new File(['reference bytes'], 'recording.wav', {type: 'audio/wav'});
  const upload = await scenario('upload', file);
  await upload.pending;
  assert.deepEqual(upload.modes, ['synthesize']);
  assert.equal(upload.calls.length, 1);
  assert.equal(upload.calls[0].url, '/api/synthesis/upload');
  const body = upload.calls[0].options.body;
  assert.ok(body instanceof FormData);
  assert.equal(body.get('file').name, 'recording.wav');
  assert.equal(await body.get('file').text(), 'reference bytes');
  const settings = JSON.parse(body.get('settings'));
  assert.equal(settings.text, 'アップロードした声で読みます。');
  assert.equal(settings.voice_id, null);
  assert.equal(settings.source_product_id, null);
  assert.equal(settings.voice_caption, undefined, 'a hidden blank description cannot block upload synthesis');
  assert.equal(upload.outputUpdates, 1);
  assert.ok(upload.$('notice').textContent.includes('先頭120秒'));

  const missing = await scenario('upload');
  await assert.rejects(missing.pending, /参照WAVを選んで/);
  assert.equal(missing.modes.length, 0, 'missing file must be caught before GPU startup');
  assert.equal(missing.calls.length, 0);

  const oversized = await scenario('upload', {size: 256 * 1024 * 1024 + 1});
  await assert.rejects(oversized.pending, /256 MiB/);
  assert.equal(oversized.modes.length, 0);

  const library = await scenario('library', file);
  await library.pending;
  assert.equal(library.calls[0].url, '/api/synthesis');
  assert.equal(JSON.parse(library.calls[0].options.body).voice_id, 'registered-voice');
  const second = new File(['second voice'], 'second.wav', {type: 'audio/wav'});
  const dialogue = await scenario('upload', [file, second], {labels: ['speaker A', 'speaker B'], transcripts: ['Aの参照文章', 'Bの参照文章'], text: '[speaker A]こんにちは。[speaker B]おはよう。'});
  await dialogue.pending;
  const multipart = dialogue.calls[0].options.body;
  assert.equal(multipart.get('file'), null);
  assert.deepEqual(multipart.getAll('files').map(file => file.name), ['recording.wav', 'second.wav']);
  const dialogueSettings = JSON.parse(multipart.get('settings'));
  assert.deepEqual(dialogueSettings.speakers, [{label: 'speaker A', reference_text: 'Aの参照文章'}, {label: 'speaker B', reference_text: 'Bの参照文章'}]);
  assert.equal(dialogueSettings.text, '[speaker A]こんにちは。[speaker B]おはよう。');
  for (const [labels, text, expected] of [
    [['', ''], '[speaker A]原稿', /ラベルを付けて/],
    [['A', 'A'], '[A]原稿', /重複/],
    [['A', 'B'], '[C]原稿', /未登録/],
    [['A', 'B'], '話者なしの原稿', /始めて/],
    [['A', 'B'], '[A][B]原稿', /空/],
  ]) {
    const invalid = await scenario('upload', [file, second], {labels, text});
    await assert.rejects(invalid.pending, expected);
    assert.equal(invalid.modes.length, 0, 'invalid dialogue must not start a model');
    assert.equal(invalid.calls.length, 0);
  }
  console.log('PASS: single/multiple uploads, ordered labels and transcripts, manuscript validation before GPU startup, registered voices');
})().catch(error => { console.error(error); process.exitCode = 1; });
