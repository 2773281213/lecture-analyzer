// 书架的公共脚本：
// - 桌面程序里（pywebview 提供 window.pywebview.api）用原生对话框选择文件、在资源管理器里打开文件夹；
//   在浏览器里打开书架时没有这些按钮，手动粘贴完整路径；
// - 课件页面在页内弹窗查看；危险操作先确认；提交后按钮变灰，避免重复提交；
// - 后台任务：顶栏显示进行中的任务数；页面上的任务显示进度，结束后刷新页面（页面上有没提交的输入时只提示）；
//   任务页完成后跳到结果。
(function () {
  "use strict";
  const FINISHED = ["done", "failed", "cancelled", "paused"]; // 已暂停的任务不再变化，停止轮询
  const STATE_CLASS = { done: "done", failed: "failed", cancelled: "skipped", paused: "skipped" };

  // ---------- 桌面程序的原生对话框 ----------
  function nativeApi() {
    return window.pywebview && window.pywebview.api ? window.pywebview.api : null;
  }

  function markNative() {
    if (nativeApi()) document.body.classList.add("has-native");
  }
  window.addEventListener("pywebviewready", markNative);
  document.addEventListener("DOMContentLoaded", markNative);

  async function pick(button) {
    const api = nativeApi();
    const target = document.getElementById(button.dataset.target);
    if (!api || !target) return;
    const mode = button.dataset.pick;
    const paths = await api.pick(mode, button.dataset.types || "");
    if (!paths || !paths.length) return;
    const existing = mode === "file" ? "" : target.value.trim();
    target.value = (existing ? existing + "\n" : "") + paths.join("\n");
    target.dispatchEvent(new Event("input", { bubbles: true }));
  }

  // ---------- 课件页面弹窗 ----------
  function showPage(link) {
    const viewer = document.getElementById("page-viewer");
    if (!viewer || typeof viewer.showModal !== "function") return false;
    document.getElementById("viewer-image").src = link.href;
    document.getElementById("viewer-image").alt = link.title || link.textContent;
    document.getElementById("viewer-title").textContent = link.title || link.textContent;
    document.getElementById("viewer-open").href = link.href;
    viewer.showModal();
    return true;
  }

  document.addEventListener("click", function (event) {
    const button = event.target.closest("[data-pick]");
    if (button) {
      event.preventDefault();
      pick(button);
      return;
    }
    const opener = event.target.closest("[data-open-path]");
    if (opener && nativeApi()) {
      event.preventDefault();
      nativeApi().open_path(opener.dataset.openPath);
      return;
    }
    const page = event.target.closest("a.page-link");
    if (page && !event.ctrlKey && !event.metaKey && showPage(page)) {
      event.preventDefault();
      return;
    }
    const dismiss = event.target.closest("[data-dismiss]");
    if (dismiss) dismiss.parentElement.remove();
  });

  // 点弹窗外面（背景）关闭
  document.addEventListener("click", function (event) {
    if (event.target.tagName === "DIALOG" && event.target.open) event.target.close();
  });

  // ---------- 表单：确认、防重复提交、记下有没有改动 ----------
  document.addEventListener("submit", function (event) {
    const form = event.target;
    if (form.dataset.confirm && !window.confirm(form.dataset.confirm)) {
      event.preventDefault();
      return;
    }
    if (form.method === "dialog") return;
    form.querySelectorAll("button:not([type=button]):not(:disabled)").forEach(function (button) {
      setTimeout(function () {
        button.disabled = true;
        button.dataset.submitting = button.textContent;
        if (button.dataset.busy) button.textContent = button.dataset.busy;
      }, 0);
    });
  });

  // 出错后按「返回」回到这一页时（浏览器缓存的页面），恢复提交时变灰的按钮
  window.addEventListener("pageshow", function (event) {
    if (!event.persisted) return;
    document.querySelectorAll("button[data-submitting]").forEach(function (button) {
      button.disabled = false;
      button.textContent = button.dataset.submitting;
      delete button.dataset.submitting;
    });
  });

  document.addEventListener("input", function (event) {
    const form = event.target.closest("form");
    if (form) form.dataset.dirty = "1";
  });

  function pageIsDirty() {
    return document.querySelector("form[data-dirty]") !== null;
  }

  // ---------- 后台任务 ----------
  async function fetchJson(url) {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) throw new Error(String(response.status));
    return response.json();
  }

  function setState(element, data) {
    if (!element) return;
    element.textContent = data.state_label;
    element.className = "chip state-" + (STATE_CLASS[data.state] || "running");
  }

  function finishedMessage(item, data) {
    const progress = item.querySelector("[data-job-progress]");
    if (!progress) return;
    progress.textContent = data.state === "failed" ? data.error : "已结束，";
    const link = document.createElement("a");
    link.href = location.pathname + location.search;
    link.textContent = "刷新页面查看结果";
    progress.appendChild(link);
  }

  function watchItem(item) {
    const id = item.dataset.jobId;
    async function poll() {
      try {
        const data = await fetchJson("/api/jobs/" + encodeURIComponent(id));
        setState(item.querySelector("[data-job-state]"), data);
        const progress = item.querySelector("[data-job-progress]");
        if (!FINISHED.includes(data.state)) {
          if (progress) progress.textContent = data.progress || "";
          setTimeout(poll, 2000);
          return;
        }
        if ("reload" in item.dataset && !pageIsDirty()) {
          location.reload();
        } else {
          finishedMessage(item, data);
        }
      } catch (error) {
        setTimeout(poll, 5000);
      }
    }
    setTimeout(poll, 1500);
  }

  function watchJobPage(box) {
    const id = box.dataset.job;
    const progress = document.getElementById("job-progress");
    const error = document.getElementById("job-error");
    async function poll() {
      try {
        const data = await fetchJson("/api/jobs/" + encodeURIComponent(id));
        setState(document.querySelector("#job-state .chip"), data);
        progress.textContent = data.state === "done" && data.summary ? data.summary : data.progress || "";
        error.textContent = data.error || "";
        error.hidden = !data.error;
        if (!FINISHED.includes(data.state)) {
          setTimeout(poll, 2000);
          return;
        }
        if (data.state !== "done") {
          if (!pageIsDirty()) location.reload(); // 换成「重试」或「继续」按钮
          return;
        }
        const actions = document.getElementById("job-actions");
        if (actions) actions.remove();
        if (data.url && data.url.startsWith("/")) {
          document.getElementById("job-link").href = data.url;
          document.getElementById("job-result").hidden = false;
          if (!(data.warnings && data.warnings.length)) setTimeout(function () { location.href = data.url; }, 1200);
          else location.reload(); // 有需要注意的事项时留在本页显示
        }
      } catch (e) {
        progress.textContent = "暂时连不上书架程序，稍后自动重试…";
        setTimeout(poll, 5000);
      }
    }
    if (!FINISHED.includes(box.dataset.state)) poll();
  }

  function watchBadge() {
    const badge = document.getElementById("jobs-badge");
    const count = document.getElementById("jobs-count");
    if (!badge || !count) return;
    async function poll() {
      try {
        const data = await fetchJson("/api/jobs");
        count.textContent = data.active;
        count.hidden = !data.active;
        badge.classList.toggle("is-busy", data.active > 0);
      } catch (e) {
        // 书架程序暂时没响应时下次再试
      }
      setTimeout(poll, 5000);
    }
    setTimeout(poll, 5000);
  }

  // ---------- 折叠的非必考内容（总复习、考前报告） ----------
  // 跳到折叠区里的知识点（目录、#k-编号 链接）时先展开；打印时全部展开，打印完恢复。
  function revealHash() {
    if (!location.hash) return;
    let target = null;
    try { target = document.querySelector(decodeURIComponent(location.hash)); } catch (e) { return; }
    const folded = target && target.closest("details.rest-points:not([open])");
    if (!folded) return;
    folded.open = true;
    target.scrollIntoView();
  }
  window.addEventListener("hashchange", revealHash);
  document.addEventListener("click", function (event) {  // 再次点同一个目录项时没有 hashchange
    if (event.target.closest('a[href^="#"]')) setTimeout(revealHash, 0);
  });
  let openedForPrint = [];
  window.addEventListener("beforeprint", function () {
    openedForPrint = Array.from(document.querySelectorAll("details.rest-points:not([open])"));
    openedForPrint.forEach(function (d) { d.open = true; });
  });
  window.addEventListener("afterprint", function () {
    openedForPrint.forEach(function (d) { d.open = false; });
    openedForPrint = [];
  });

  document.addEventListener("DOMContentLoaded", function () {
    revealHash();
    document.querySelectorAll("[data-job-id]").forEach(watchItem);
    const box = document.getElementById("job");
    if (box) watchJobPage(box);
    watchBadge();
    document.querySelectorAll("dialog[data-open]").forEach(function (dialog) {
      if (typeof dialog.showModal === "function") dialog.showModal();
      else dialog.setAttribute("open", "");
    });
  });
})();
