// The library shares the chat runtime; switching modes unloads the previous model.
async function libraryFetch(path, options = {}) {
  const res = await fetch(`/api/voice/library${path}`, options);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail ?? `声ライブラリ: ${res.status}`);
  }
  return res;
}
function libraryJSON(method, body) {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}
function updateLibraryRuntime(runtime = state.status?.voice) {
  const ready = runtime?.enabled && runtime.engine === "irodori" && runtime.ready;
  const error = Object.values(runtime?.errors ?? {}).join(" / ");
  $("libraryRuntime").textContent = error || (!runtime?.enabled ? "音声はOFFです。" :
    runtime.engine !== "irodori" ? "AivisSpeech / VOICEVOXを利用中です。" :
    runtime.starting ? "Irodoriをロード・ウォームアップ中…" :
    ready ? `Irodori ${runtime.mode === "design" ? "声作成（Large GPU / RF40）" : "会話（MF4）"}を利用できます。` : "Irodoriの準備が完了していません。");
  $("generateCandidates").disabled = !!state.libraryBusy || !ready || runtime.mode !== "design";
  for (const button of $("libraryItems").querySelectorAll("[data-preview]")) {
    button.disabled = !!state.libraryBusy || !ready || runtime.mode !== "chat";
  }
  for (const id of ["startDesign", "startChatVoice"]) $(id).disabled = !!state.voiceChanging || !!state.libraryBusy;
  $("libraryOff").disabled = !!state.voiceChanging || !runtime?.enabled;
  if (runtime?.engine === "irodori" && runtime.mode === "design") {
    $("voiceStatus").textContent = error || (ready ? "声作成モードです。会話モードに切り替えると読み上げできます。" : "声作成モデルをロード中…");
  } else if (error) $("voiceStatus").textContent = error;
}
async function libraryAction(action, message = "処理中…") {
  if (state.libraryBusy) return;
  state.libraryBusy = true;
  $("libraryStatus").textContent = message;
  updateLibraryRuntime();
  try { await action(); }
  catch (e) { $("libraryStatus").textContent = e.message; }
  finally { state.libraryBusy = false; updateLibraryRuntime(); }
}
function stopLibraryAudio() {
  $("libraryPreview").pause();
  $("libraryPreview").removeAttribute("src");
  $("libraryPreview").hidden = true;
  if (state.libraryAudioURL) URL.revokeObjectURL(state.libraryAudioURL);
  state.libraryAudioURL = null;
  for (const audio of $("libraryItems").querySelectorAll("audio")) audio.pause();
}
async function loadLibrary() {
  const data = await (await libraryFetch("")).json();
  state.libraryItems = data.items;
  if (!$("candidateText").value) $("candidateText").value = data.default_text;
  if (!$("previewVoicePreset").options.length) {
    for (const preset of data.presets) $("previewVoicePreset").add(new Option(preset.name, preset.id));
  }
  renderLibrary();
}
function renderLibrary() {
  stopLibraryAudio();
  $("libraryItems").replaceChildren();
  const items = (state.libraryItems ?? []).filter(item => !$("favoritesOnly").checked || item.favorite);
  if (!items.length) $("libraryItems").textContent = "保存済みの声はありません。候補を生成するかWAVを登録してください。";
  for (const item of items) {
    const card = document.createElement("article");
    card.className = "voice-card";
    const duration = item.references.reduce((sum, ref) => sum + ref.seconds, 0);
    // Only local immutable IDs form URLs; captions and names are escaped text.
    card.innerHTML = `<b>${esc(item.name)}</b> · ${item.kind === "voice" ? "登録済み" : "候補"}
      <div><small>${duration.toFixed(1)}秒 · 参照${item.references.length}件 · seed ${item.provenance.seed ?? "—"}</small></div>
      <p>${esc(item.provenance.caption ?? item.provenance.usage_notes ?? "取り込み音声")}</p>
      <audio controls preload="none" src="/api/voice/library/${item.id}/audio"></audio>
      <div class="actions">
        <button data-favorite type="button">${item.favorite ? "★ お気に入り" : "☆ お気に入り"}</button>
        ${item.kind !== "voice" ? '<button data-register type="button">この声を登録</button>' : '<button data-select type="button">会話で選ぶ</button>'}
        <button data-preview type="button">別の文章で試聴（MF4）</button>
        <a href="/api/voice/library/${item.id}/audio">WAVを保存</a>
        <a href="/api/voice/library/${item.id}/metadata" target="_blank" rel="noopener">生成条件・SHA256</a>
      </div>
      <div class="actions"><input data-name type="text" maxlength="80" aria-label="声の表示名" value="${esc(item.name)}"><button data-rename type="button">名前を保存</button><button data-delete type="button">削除</button></div>
      ${item.kind === "voice" ? '<details><summary>同じ話者の補助参照を追加</summary><input data-reference type="file" accept="audio/wav,.wav"><label class="opt"><input data-same-speaker type="checkbox">同じ話者のクリップです（別seedの候補は混ぜません）</label><button data-append type="button">参照を追加</button></details>' : ''}`;
    const update = async (body) => {
      await libraryFetch(`/${item.id}`, libraryJSON("PATCH", body));
      await loadLibrary();
      if (state.voiceEnabled) refreshVoices();
      $("libraryStatus").textContent = "保存しました。";
    };
    card.querySelector("[data-favorite]").onclick = () => libraryAction(() => update({ favorite: !item.favorite }));
    card.querySelector("[data-register]")?.addEventListener("click", () => libraryAction(() => update({ register: true })));
    card.querySelector("[data-rename]").onclick = () => libraryAction(() => update({ name: card.querySelector("[data-name]").value.trim() }));
    card.querySelector("[data-delete]").onclick = () => {
      if (confirm(`「${item.name}」と参照音声を削除しますか？`)) libraryAction(async () => {
        stopSpeech();
        await libraryFetch(`/${item.id}`, { method: "DELETE" });
        await loadLibrary();
        if (state.voiceEnabled) refreshVoices();
        $("libraryStatus").textContent = "削除しました。";
      });
    };
    card.querySelector("[data-select]")?.addEventListener("click", () => libraryAction(async () => {
      await startLibraryMode("chat");
      state.desiredVoice = `irodori:${item.id}`;
      $("libraryStatus").textContent = "会話モードでこの声を選びます。モデルの準備が完了したら会話できます。";
    }));
    card.querySelector("[data-preview]").onclick = () => libraryAction(async () => {
      stopSpeech();
      stopLibraryAudio();
      const res = await libraryFetch(`/${item.id}/preview`, libraryJSON("POST", {
        text: $("previewVoiceText").value.trim(), preset: $("previewVoicePreset").value,
      }));
      const blob = await res.blob();
      if (!$("voiceLibrary").open) return;
      state.libraryAudioURL = URL.createObjectURL(blob);
      $("libraryPreview").src = state.libraryAudioURL;
      $("libraryPreview").hidden = false;
      const metrics = JSON.parse(res.headers.get("X-TTS-Metrics") ?? "{}");
      $("libraryStatus").textContent = `合成 ${metrics.elapsed_seconds?.toFixed(2)}秒 / 音声 ${metrics.audio_seconds?.toFixed(2)}秒 / RTF ${metrics.rtf?.toFixed(2)}`;
      await $("libraryPreview").play();
    }, "参照を準備して合成中…");
    card.querySelector("[data-append]")?.addEventListener("click", () => libraryAction(async () => {
      const file = card.querySelector("[data-reference]").files[0];
      if (!file || !card.querySelector("[data-same-speaker]").checked) throw new Error("同じ話者のWAVを選んで確認してください");
      const form = new FormData(); form.append("file", file); form.append("same_speaker", "true");
      await libraryFetch(`/${item.id}/references`, { method: "POST", body: form });
      await loadLibrary();
      $("libraryStatus").textContent = "補助参照を追加しました。次の合成で参照を再準備します。";
    }));
    $("libraryItems").append(card);
  }
  updateLibraryRuntime();
}
async function startLibraryMode(mode) {
  if (state.voiceChanging) return;
  stopLibraryAudio();
  $("voiceEngine").value = "irodori";
  $("voiceMode").value = mode;
  $("voiceMode").hidden = false;
  $("speak").checked = true;
  await toggleVoice();
}
$("voiceLibraryBtn").onclick = () => {
  $("voiceLibrary").showModal();
  loadLibrary().catch(e => { $("libraryStatus").textContent = e.message; });
};
$("closeLibrary").onclick = () => $("voiceLibrary").close();
$("voiceLibrary").addEventListener("close", stopLibraryAudio);
$("reloadLibrary").onclick = () => libraryAction(loadLibrary);
$("favoritesOnly").onchange = renderLibrary;
$("startDesign").onclick = () => startLibraryMode("design");
$("startChatVoice").onclick = () => startLibraryMode("chat");
$("libraryOff").onclick = () => { stopLibraryAudio(); $("speak").checked = false; toggleVoice(); };
$("candidateForm").onsubmit = e => {
  e.preventDefault();
  libraryAction(async () => {
    const seed = $("candidateSeed").value;
    await libraryFetch("/candidates", libraryJSON("POST", {
      caption: $("candidateCaption").value.trim(), text: $("candidateText").value.trim(),
      name: $("candidateName").value.trim(), seed: seed === "" ? null : Number(seed),
      count: Number($("candidateCount").value),
    }));
    await loadLibrary();
    $("libraryStatus").textContent = "候補を保存しました。試聴して、使いたい声を登録してください。";
  }, "Large GPU / RF40で候補を順番に生成中… 保存済みの候補は一覧の更新で確認できます。");
};
$("importVoiceForm").onsubmit = e => {
  e.preventDefault();
  libraryAction(async () => {
    const form = new FormData();
    form.append("file", $("importVoiceFile").files[0]);
    form.append("name", $("importVoiceName").value.trim());
    form.append("source", $("importVoiceSource").value.trim());
    await libraryFetch("/import", { method: "POST", body: form });
    await loadLibrary();
    if (state.voiceEnabled) refreshVoices();
    $("libraryStatus").textContent = "声を登録しました。";
  });
};
