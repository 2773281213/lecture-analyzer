// 讲的阅读页：一讲可以由多节音视频组成，按整讲时间统一跳转、连续播放，多节的讲用一条整讲进度条；
// 播放时高亮当前转写句；地址带 ?t=毫秒 时打开页面即定位到该时间（跨讲的联系用它跳转）；渲染公式
(() => {
  const player = document.getElementById("player");
  const dataTag = document.getElementById("parts-data");
  const parts = dataTag ? JSON.parse(dataTag.textContent) : [];
  const lines = Array.from(document.querySelectorAll(".line[data-t]"));
  let partIndex = 0;
  let current = null;
  let pending = null; // 等文件加载完才能定位时的目标（整讲时间）：加载完之前按它显示，换节时不闪回这一节的开头

  // 整讲时间落在哪一节：最后一个起点不晚于该时间的节
  const partAt = (ms) => {
    let index = 0;
    parts.forEach((part, i) => { if (part.offset <= ms) index = i; });
    return index;
  };

  const loadPart = (index) => {
    partIndex = index;
    player.src = parts[index].src;
  };

  const seek = (ms, play) => {
    if (!player || !parts.length) return;
    const index = partAt(ms);
    const go = () => {
      player.currentTime = Math.max(0, (ms - parts[index].offset) / 1000);
      pending = null;
      if (play) player.play().catch(() => {}); // 浏览器禁止自动播放时，至少已跳到对应位置
    };
    if (index === partIndex && player.readyState >= 1) {
      go();
      return;
    }
    // 要等这一节的文件加载完才能定位
    pending = ms;
    if (index !== partIndex) loadPart(index);
    player.addEventListener("loadedmetadata", go, { once: true });
  };

  document.addEventListener("click", (event) => {
    const target = event.target.closest("[data-t]");
    if (!target || !player) return;
    event.preventDefault();
    seek(Number(target.dataset.t), true);
  });

  // 找到最后一句开始时间不晚于当前播放位置的转写句
  const lineAt = (ms) => {
    let low = 0;
    let high = lines.length - 1;
    let found = null;
    while (low <= high) {
      const mid = (low + high) >> 1;
      if (Number(lines[mid].dataset.t) <= ms) {
        found = lines[mid];
        low = mid + 1;
      } else {
        high = mid - 1;
      }
    }
    return found;
  };

  const highlight = (ms) => {
    const line = lineAt(ms);
    if (line === current) return;
    current?.classList.remove("is-current");
    line?.classList.add("is-current");
    current = line;
    const box = line?.closest(".lines");
    if (line && box && box.closest("details")?.open) {
      box.scrollTo({ top: line.offsetTop - box.clientHeight / 3, behavior: "smooth" });
    }
  };

  // 当前播放位置（整讲时间）
  const position = () => pending ?? parts[partIndex].offset + player.currentTime * 1000;

  if (player && parts.length) {
    player.addEventListener("timeupdate", () => {
      if (lines.length) highlight(position());
    });
    // 一节放完自动接着放下一节
    player.addEventListener("ended", () => {
      if (partIndex + 1 >= parts.length) return;
      loadPart(partIndex + 1);
      player.play().catch(() => {});
    });
    const start = new URLSearchParams(location.search).get("t");
    if (start !== null && /^\d+$/.test(start)) {
      seek(Number(start), false);
      highlight(Number(start));
    }
  }

  // 与 timecode.format_ms 一致：不到一小时 mm:ss，否则 h:mm:ss
  const formatMs = (ms) => {
    const seconds = Math.max(0, Math.floor(ms / 1000));
    const pad = (n) => String(n).padStart(2, "0");
    const rest = `${pad(Math.floor((seconds % 3600) / 60))}:${pad(seconds % 60)}`;
    return seconds >= 3600 ? `${Math.floor(seconds / 3600)}:${rest}` : rest;
  };

  // 多节的讲：整讲一条进度条、整讲时间，换节由脚本完成（M4 验收：原生控件只显示当前这一节，像被分成了几段）
  const controls = document.getElementById("player-controls");
  if (player && controls && parts.length > 1) {
    const playButton = controls.querySelector(".pc-play");
    const now = controls.querySelector(".pc-now");
    const bar = controls.querySelector(".pc-seek");
    const rate = controls.querySelector(".pc-rate");
    const mute = controls.querySelector(".pc-mute");
    const full = controls.querySelector(".pc-full");
    let dragging = false;
    const toggle = () => (player.paused ? player.play().catch(() => {}) : player.pause());
    const show = () => {
      const ms = position();
      now.textContent = formatMs(ms);
      if (!dragging) bar.value = ms;
    };

    player.controls = false;
    controls.hidden = false;
    player.addEventListener("click", toggle);
    playButton.addEventListener("click", toggle);
    player.addEventListener("play", () => { playButton.textContent = "暂停"; });
    player.addEventListener("pause", () => { playButton.textContent = "播放"; });
    player.addEventListener("timeupdate", show);
    // 拖动时只预览时间，松开后再跳转（跨节时换文件）
    bar.addEventListener("input", () => {
      dragging = true;
      now.textContent = formatMs(Number(bar.value));
    });
    bar.addEventListener("change", () => {
      dragging = false;
      seek(Number(bar.value), !player.paused);
      show();
    });
    // 换节时浏览器会把倍速重置为 defaultPlaybackRate，两个一起设才能保持
    rate.addEventListener("change", () => {
      player.defaultPlaybackRate = Number(rate.value);
      player.playbackRate = Number(rate.value);
    });
    mute.addEventListener("click", () => {
      player.muted = !player.muted;
      mute.textContent = player.muted ? "取消静音" : "静音";
    });
    if (full) {
      const frame = controls.closest(".player-frame");
      full.addEventListener("click", () => {
        (document.fullscreenElement ? document.exitFullscreen() : frame.requestFullscreen()).catch(() => {});
      });
      document.addEventListener("fullscreenchange", () => {
        full.textContent = document.fullscreenElement ? "退出全屏" : "全屏";
      });
    }
    show();
  }

  // KaTeX 的自动渲染脚本加载完后调用；离线时公式保持 LaTeX 原文。
  // 渲染整个 <main>：报告、问答、资料页的 <main> 不是 .page（M4 验收：这三页的公式原来都没渲染）
  window.renderMath = () => {
    if (!window.renderMathInElement) return;
    window.renderMathInElement(document.querySelector("main"), {
      delimiters: [
        { left: "$$", right: "$$", display: true },
        { left: "$", right: "$", display: false },
      ],
      throwOnError: false,
    });
  };
})();
