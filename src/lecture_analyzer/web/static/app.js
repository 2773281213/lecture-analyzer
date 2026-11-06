// 书架的公共脚本：
// - 桌面程序里（pywebview 提供 window.pywebview.api）用原生对话框选择文件、在资源管理器里打开文件夹；
//   在浏览器里打开书架时没有这些按钮，手动粘贴完整路径；
// - 课件页面在页内弹窗查看；危险操作先确认；提交后按钮变灰，避免重复提交；
// - 勾选后批量操作的表单（如批量删除资料）：表头全选，一个都没勾时提示，确认文字里显示勾选的个数；
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
    const duplicates = event.target.closest("[data-check-duplicates]");  // 资料表：勾选来源重复的几份
    if (duplicates && duplicates.form) {
      duplicates.form.querySelectorAll("input[data-duplicate]").forEach(function (box) { box.checked = true; });
      duplicates.form.dataset.dirty = "1";
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
  // data-need-checked 为批量操作的复选框名：一个都没勾时提示 data-empty；确认文字里的 {n} 换成勾选的个数
  function checkedCount(form) {
    return form.querySelectorAll('input[type=checkbox][name="' + form.dataset.needChecked + '"]:checked').length;
  }

  document.addEventListener("submit", function (event) {
    const form = event.target;
    const count = form.dataset.needChecked ? checkedCount(form) : null;
    if (count === 0) {
      event.preventDefault();
      window.alert(form.dataset.empty || "请先勾选");
      return;
    }
    let message = form.dataset.confirm || "";
    if (count !== null) message = message.replace("{n}", String(count));
    if (message && !window.confirm(message)) {
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

  // 表头的全选框：data-check-all 为同一表单里要一起勾选的复选框名
  document.addEventListener("change", function (event) {
    const all = event.target.closest("[data-check-all]");
    const form = all && all.closest("form");
    if (!form) return;
    form.querySelectorAll('input[type=checkbox][name="' + all.dataset.checkAll + '"]').forEach(function (box) {
      box.checked = all.checked;
    });
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

  // 状态栏：第一个进行中的任务和进度；书架首页：最近的几个任务
  function showJobs(data) {
    const status = document.getElementById("status-job");
    const text = document.getElementById("status-job-text");
    if (status && text) {
      const running = data.jobs.filter(function (job) { return job.active; });
      const first = running.find(function (job) { return job.state === "running"; }) || running[0];
      status.classList.toggle("is-idle", !first);
      text.textContent = first
        ? "#" + first.id + " " + first.label + (first.progress ? " · " + first.progress : "") +
          (running.length > 1 ? "（共 " + running.length + " 个任务）" : "")
        : "没有进行中的后台任务";
      status.href = first ? "/jobs/" + first.id : "/jobs";
    }
    const recent = document.getElementById("recent-jobs");
    if (!recent) return;
    recent.replaceChildren();
    data.jobs.slice(0, 5).forEach(function (job) {
      const row = document.createElement("tr");
      const id = document.createElement("td");
      id.className = "num muted-cell";
      id.style.width = "56px";
      id.textContent = "#" + job.id;
      const label = document.createElement("td");
      const link = document.createElement("a");
      link.href = "/jobs/" + job.id;
      link.textContent = job.label;
      label.appendChild(link);
      const state = document.createElement("td");
      const chip = document.createElement("span");
      setState(chip, job);
      state.appendChild(chip);
      if (job.active && job.progress) state.append(" " + job.progress);
      const time = document.createElement("td");
      time.className = "num muted-cell";
      time.style.whiteSpace = "nowrap";
      time.textContent = (job.updated_at || job.created_at || "").replace("T", " ").slice(5, 16);
      row.append(id, label, state, time);
      recent.appendChild(row);
    });
    if (!data.jobs.length) {
      const row = recent.insertRow();
      row.insertCell().textContent = "还没有后台任务";
      row.cells[0].className = "muted";
    }
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
        showJobs(data);
      } catch (e) {
        // 书架程序暂时没响应时下次再试
      }
      setTimeout(poll, 5000);
    }
    // 有任务在跑、或页面上有任务列表时马上读一次（状态栏显示具体是哪个任务）
    setTimeout(poll, document.getElementById("recent-jobs") || !count.hidden ? 0 : 5000);
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

// 工作台：主题、侧栏、列表选中后在右栏显示详情、筛选、页签、侧栏目录跟随滚动。
// 界面偏好记在 Cookie 里（web/themes.py 读取）：Cookie 不分端口，桌面程序每次启动端口不同也能记住。
(function () {
  "use strict";
  const root = document.documentElement;
  const YEAR = 365 * 24 * 3600;

  function remember(name, value) {
    document.cookie = name + "=" + encodeURIComponent(value) + "; max-age=" + YEAR + "; path=/; SameSite=Lax";
  }

  // ---------- 主题 ----------
  const menu = document.getElementById("theme-menu");
  const menuButton = document.getElementById("theme-button");
  const systemDark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;

  function syncPrefs() {
    const chosen = root.getAttribute("data-chosen");
    document.querySelectorAll("[data-set-theme]").forEach(function (option) {
      option.setAttribute("aria-checked", option.dataset.setTheme === chosen ? "true" : "false");
    });
    const option = document.querySelector('[data-set-theme="' + chosen + '"]');
    if (option) {
      document.querySelectorAll("[data-theme-name]").forEach(function (el) { el.textContent = option.dataset.name; });
    }
    document.querySelectorAll('[data-pref="autodark"]').forEach(function (box) { box.checked = root.hasAttribute("data-autodark"); });
    document.querySelectorAll('[data-pref="glass"]').forEach(function (box) { box.checked = root.getAttribute("data-glass") !== "off"; });
  }

  function switching(change) {
    root.classList.add("theme-switching");  // 换主题时不播放过渡动画，免得各处颜色先后变化
    change();
    if (window.lecApplyTheme) window.lecApplyTheme();
    syncPrefs();
    requestAnimationFrame(function () {
      requestAnimationFrame(function () { root.classList.remove("theme-switching"); });
    });
  }

  function setTheme(id) {
    switching(function () {
      root.setAttribute("data-chosen", id);
      remember("lec_theme", id);
      // 系统是深色、又开着「跟随系统」时选了浅色主题会看不到变化：明确选的主题优先，关掉跟随
      if (id !== "night" && root.hasAttribute("data-autodark") && systemDark && systemDark.matches) {
        root.removeAttribute("data-autodark");
        remember("lec_autodark", "0");
      }
    });
  }

  function setPref(name, on) {
    switching(function () {
      if (name === "autodark") {
        root.toggleAttribute("data-autodark", on);
        remember("lec_autodark", on ? "1" : "0");
      } else if (name === "glass") {
        if (on) root.removeAttribute("data-glass");
        else root.setAttribute("data-glass", "off");
        remember("lec_glass", on ? "1" : "0");
      }
    });
  }

  function openMenu(open) {
    if (!menu) return;
    menu.hidden = !open;
    if (menuButton) menuButton.setAttribute("aria-expanded", open ? "true" : "false");
    if (open) {
      const current = menu.querySelector('[aria-checked="true"]') || menu.querySelector("[data-set-theme]");
      if (current) current.focus();
    }
  }

  // ---------- 侧栏：宽窗口里显示 / 隐藏（记住），窄窗口里临时展开 ----------
  const app = document.getElementById("app");
  const narrow = window.matchMedia ? window.matchMedia("(max-width: 1080px)") : null;

  function toggleSide() {
    if (!app) return;
    if (narrow && narrow.matches) {
      app.classList.toggle("side-peek");
      return;
    }
    const off = !app.classList.contains("side-off");
    app.classList.toggle("side-off", off);
    remember("lec_side", off ? "0" : "1");
  }

  // ---------- 列表选中一行，右栏显示它的详情 ----------
  function select(row) {
    const table = row.closest("table");
    if (!table) return;
    table.querySelectorAll("tr.is-selected").forEach(function (r) { r.classList.remove("is-selected"); });
    row.classList.add("is-selected");
    const id = row.dataset.select;
    document.querySelectorAll(".insp-panel[data-panel]").forEach(function (panel) {
      panel.hidden = panel.dataset.panel !== id;
    });
  }

  // ---------- 页签（右栏的转写 / 考点信号 / 本讲信息） ----------
  function showTab(tab) {
    const list = tab.closest('[role="tablist"]');
    const scope = list.parentElement;
    list.querySelectorAll('[role="tab"]').forEach(function (t) {
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    scope.querySelectorAll("[data-pane]").forEach(function (pane) { pane.hidden = pane.dataset.pane !== tab.dataset.tab; });
  }

  function filterLectures(button) {
    const wanted = button.dataset.state;
    button.parentElement.querySelectorAll("[data-state]").forEach(function (b) {
      b.setAttribute("aria-pressed", b === button ? "true" : "false");
    });
    document.querySelectorAll("#lecture-table tbody tr[data-state]").forEach(function (row) {
      row.hidden = Boolean(wanted) && row.dataset.state !== wanted;
    });
  }

  function setShelfView(button) {
    const shelf = document.getElementById("shelf");
    if (shelf) shelf.className = shelf.className.replace(/shelf-view-\w+/, "shelf-view-" + button.dataset.shelfView);
    document.querySelectorAll("[data-shelf-view]").forEach(function (b) {
      b.setAttribute("aria-pressed", b === button ? "true" : "false");
    });
    remember("lec_shelf_view", button.dataset.shelfView);
  }

  document.addEventListener("click", function (event) {
    const target = event.target;
    if (target.closest("#theme-button, [data-open-theme]")) {
      openMenu(menu.hidden);
      return;
    }
    const option = target.closest("[data-set-theme]");
    if (option) {
      setTheme(option.dataset.setTheme);
      return;
    }
    if (menu && !menu.hidden && !target.closest("#theme-menu")) openMenu(false);
    if (target.closest("#side-toggle")) {
      toggleSide();
      return;
    }
    if (app && app.classList.contains("side-peek") && !target.closest("#side")) app.classList.remove("side-peek");
    const view = target.closest("[data-shelf-view]");
    if (view) {
      setShelfView(view);
      return;
    }
    const stateButton = target.closest("[data-state-filter] [data-state]");
    if (stateButton) {
      filterLectures(stateButton);
      return;
    }
    const tab = target.closest('[role="tab"][data-tab]');
    if (tab) {
      showTab(tab);
      return;
    }
    const opener = target.closest("[data-open-details]");
    if (opener) {
      const details = document.getElementById(opener.dataset.openDetails);
      if (details) details.open = true;
    }
    const row = target.closest("tr[data-select]");
    if (row && !target.closest("a, button, input, select, form, label")) select(row);
  });

  // 双击一行打开它（课程、讲）
  document.addEventListener("dblclick", function (event) {
    const row = event.target.closest("tr[data-select]");
    if (!row || event.target.closest("a, button, input, select, form")) return;
    const link = row.querySelector("a[href]");
    if (link) location.href = link.href;
  });

  document.addEventListener("change", function (event) {
    const pref = event.target.closest("[data-pref]");
    if (pref) setPref(pref.dataset.pref, pref.checked);
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      if (menu && !menu.hidden) {
        openMenu(false);
        if (menuButton) menuButton.focus();
      }
      if (app) app.classList.remove("side-peek");
      return;
    }
    // 列表里用上下键换选中的行
    if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
    if (event.target.closest("input, select, textarea, audio, video, #theme-menu")) return;
    const current = document.querySelector("tr[data-select].is-selected");
    if (!current) return;
    const step = function (row) { return event.key === "ArrowDown" ? row.nextElementSibling : row.previousElementSibling; };
    let next = step(current);
    while (next && (next.hidden || !next.dataset.select)) next = step(next);
    if (!next) return;
    event.preventDefault();
    select(next);
    next.scrollIntoView({ block: "nearest" });
  });

  // ---------- 书架首页：筛选课程 ----------
  document.addEventListener("input", function (event) {
    if (event.target.id !== "course-filter") return;
    const words = event.target.value.trim().toLowerCase();
    document.querySelectorAll("[data-filter]").forEach(function (el) {
      el.hidden = Boolean(words) && !el.dataset.filter.toLowerCase().includes(words);
    });
    document.querySelectorAll("[data-course-item]").forEach(function (el) {
      el.hidden = Boolean(words) && !el.dataset.courseItem.toLowerCase().includes(words);
    });
  });

  // ---------- 侧栏目录跟随滚动：标出正在看的部分 ----------
  function spy() {
    const main = document.getElementById("main");
    const links = Array.from(document.querySelectorAll('[data-spy] a[href^="#"]'));
    if (!main || !links.length || !window.IntersectionObserver) return;
    const targets = links.map(function (link) {
      try { return document.querySelector(decodeURIComponent(link.getAttribute("href"))); } catch (e) { return null; }
    });
    const visible = new Set();
    function mark() {
      // 离内容区顶部最近的可见部分；知识点卡片优先标在侧栏的子项上
      let index = -1;
      let best = Infinity;
      targets.forEach(function (el, i) {
        if (!el || !visible.has(el)) return;
        const score = Math.abs(el.getBoundingClientRect().top - 90) - (links[i].classList.contains("side-sub") ? 40 : 0);
        if (score < best) {
          best = score;
          index = i;
        }
      });
      if (index >= 0) links.forEach(function (link, i) { link.classList.toggle("is-on", i === index); });
    }
    const observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (entry.isIntersecting) visible.add(entry.target);
        else visible.delete(entry.target);
      });
      mark();
    }, { root: main, rootMargin: "-60px 0px -45% 0px" });
    targets.forEach(function (el) { if (el) observer.observe(el); });
  }

  document.addEventListener("DOMContentLoaded", function () {
    syncPrefs();
    spy();
    // 窗口从窄变宽时收起临时展开的侧栏
    if (narrow && narrow.addEventListener) {
      narrow.addEventListener("change", function () { if (app) app.classList.remove("side-peek"); });
    }
  });
})();
