const API_BASE = "/api/v1";

const $ = (id) => document.getElementById(id);

async function request(path, options = {}) {
  const response = await fetch(API_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    throw new Error(`${response.status}: ${detail}`);
  }
  return body;
}

function element(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  Object.entries(attrs).forEach(([key, value]) => {
    if (key === "text") node.textContent = value;
    else node.setAttribute(key, value);
  });
  children.forEach((child) => node.append(child));
  return node;
}

function notice(container, message, kind) {
  container.replaceChildren(element("div", { class: `notice notice--${kind}`, text: message }));
}

function renderResults(container, results) {
  if (!results.length) {
    container.append(element("p", { class: "meta", text: "Tidak ada hasil." }));
    return;
  }
  results.forEach((result, index) => {
    const source = result.metadata?.source ?? "tanpa source";
    const score = typeof result.score === "number" ? result.score.toFixed(3) : "-";
    container.append(
      element("details", { class: "result" }, [
        element("summary", {}, [
          element("span", { text: `${index + 1}. ${source}` }),
          element("span", { class: "score", text: score }),
        ]),
        element("pre", { text: result.text ?? "" }),
      ]),
    );
  });
}

const TIER_LABELS = { human: "human-reviewed", machine: "machine", unverified: "unverified" };

function renderConcepts(container, concepts) {
  if (!concepts.length) return;
  container.append(element("p", { class: "meta", text: "Konsep terkurasi" }));
  concepts.forEach((concept, index) => {
    const tier = concept.verified_tier ?? "unverified";
    container.append(
      element("div", { class: "concept" }, [
        element("span", { text: `K${index + 1}. ${concept.title}` }),
        element("span", { class: "concept__path", text: concept.path }),
        element("span", { class: `badge badge--${tier}`, text: TIER_LABELS[tier] ?? tier }),
      ]),
    );
  });
}

async function withBusy(form, task) {
  const button = form.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    await task();
  } finally {
    button.disabled = false;
  }
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      document.querySelectorAll(".panel").forEach((panel) => {
        panel.classList.toggle("is-active", panel.id === `panel-${tab.dataset.tab}`);
      });
    });
  });
}

async function checkHealth() {
  const badge = $("health");
  try {
    const body = await request("/health");
    badge.textContent = `API aktif · v${body.version}`;
    badge.className = "health health--ok";
  } catch {
    badge.textContent = "API tidak terjangkau";
    badge.className = "health health--down";
  }
}

function setupAnswer() {
  const form = $("answer-form");
  const output = $("answer-output");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    withBusy(form, async () => {
      notice(output, "Menyusun jawaban…", "ok");
      try {
        const body = await request("/answer", {
          method: "POST",
          body: JSON.stringify({ query: $("answer-query").value, top_k: Number($("answer-topk").value) }),
        });
        output.replaceChildren(
          element("div", { class: "answer", text: body.answer }),
          element("p", { class: "meta", text: `${body.provider} · ${body.model} · ${body.count} sumber` }),
        );
        renderConcepts(output, body.concepts ?? []);
        renderResults(output, body.sources ?? []);
      } catch (error) {
        notice(output, error.message, "error");
      }
    });
  });
}

function setupSearch() {
  const form = $("search-form");
  const output = $("search-output");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    withBusy(form, async () => {
      notice(output, "Mencari…", "ok");
      try {
        const body = await request("/query", {
          method: "POST",
          body: JSON.stringify({
            query: $("search-query").value,
            top_k: Number($("search-topk").value),
            hybrid: $("search-hybrid").checked,
          }),
        });
        output.replaceChildren();
        if (!body.use_rag) {
          output.append(element("p", { class: "meta", text: "Query tidak dirutekan ke RAG oleh intent classifier." }));
          return;
        }
        output.append(element("p", { class: "meta", text: `${body.count} hasil` }));
        renderResults(output, body.results ?? []);
      } catch (error) {
        notice(output, error.message, "error");
      }
    });
  });
}

function setupUpsert() {
  const form = $("upsert-form");
  const output = $("upsert-output");
  $("upsert-file").addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;
    $("upsert-text").value = await file.text();
    if (!$("upsert-source").value) $("upsert-source").value = file.name;
  });
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    withBusy(form, async () => {
      try {
        const body = await request("/documents/upsert", {
          method: "POST",
          body: JSON.stringify({ text: $("upsert-text").value, source: $("upsert-source").value }),
        });
        if (body.status !== "success") throw new Error(body.message ?? "Upsert gagal");
        notice(output, `Tersimpan: ${body.chunks_stored} chunk dari ${body.source}`, "ok");
      } catch (error) {
        notice(output, error.message, "error");
      }
    });
  });
}

function setupDelete() {
  const form = $("delete-form");
  const output = $("delete-output");
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    const source = $("delete-source").value;
    if (!confirm(`Hapus semua chunk dari "${source}"?`)) return;
    withBusy(form, async () => {
      try {
        const body = await request(`/documents?source=${encodeURIComponent(source)}`, { method: "DELETE" });
        notice(output, `Terhapus: ${body.chunks_deleted} chunk dari ${body.source}`, "ok");
      } catch (error) {
        notice(output, error.message, "error");
      }
    });
  });
}

setupTabs();
setupAnswer();
setupSearch();
setupUpsert();
setupDelete();
checkHealth();
