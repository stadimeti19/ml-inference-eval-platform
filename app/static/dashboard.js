/* Dashboard interactions. Server-rendered data stays in dashboard.html. */
function switchTab(name) {
  document.querySelectorAll(".tab").forEach((tab) => tab.classList.remove("active"));
  document.querySelectorAll(".nav-item").forEach((item) => item.classList.remove("active"));
  document.getElementById(`tab-${name}`).classList.add("active");
  document.querySelector(`[data-tab="${name}"]`).classList.add("active");
}

function toast(message, ok) {
  const element = document.getElementById("toast");
  element.textContent = message;
  element.className = `toast show ${ok ? "toast-ok" : "toast-err"}`;
  setTimeout(() => { element.className = "toast"; }, 3000);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || "Request failed");
  return data;
}

async function promote(modelName, version) {
  if (!confirm(`Promote ${modelName}@${version} to production?`)) return;
  try {
    await postJson("/dashboard/api/promote", { model_name: modelName, model_version: version });
    toast(`Promoted ${version} to prod`, true);
    setTimeout(() => location.reload(), 800);
  } catch (error) { toast(error.message, false); }
}

async function rollback(modelName) {
  if (!confirm(`Rollback ${modelName} to previous version?`)) return;
  try {
    const data = await postJson("/dashboard/api/rollback", { model_name: modelName });
    toast(`Rolled back to ${data.model_version || "previous"}`, true);
    setTimeout(() => location.reload(), 800);
  } catch (error) { toast(error.message, false); }
}

async function createSlo() {
  const name = document.getElementById("slo-name").value;
  const model = document.getElementById("slo-model").value;
  const p95 = document.getElementById("slo-p95").value;
  const accuracy = document.getElementById("slo-acc").value;
  if (!name || !model) { toast("Name and model required", false); return; }

  const constraints = {};
  if (p95) constraints.p95_ms_max = parseFloat(p95);
  if (accuracy) constraints.accuracy_min = parseFloat(accuracy);
  if (Object.keys(constraints).length === 0) { toast("Add at least one constraint", false); return; }

  try {
    await postJson("/slo/policies", { name, model_name: model, constraints });
    toast("Policy created", true);
    setTimeout(() => location.reload(), 800);
  } catch (error) { toast(error.message, false); }
}

async function deleteSlo(name) {
  if (!confirm(`Delete policy "${name}"?`)) return;
  try {
    const response = await fetch(`/slo/policies/${encodeURIComponent(name)}`, { method: "DELETE" });
    if (!response.ok) throw new Error("Delete failed");
    toast("Deleted", true);
    setTimeout(() => location.reload(), 800);
  } catch (error) { toast(error.message, false); }
}

async function runSloCheck() {
  const model = document.getElementById("chk-model").value;
  const version = document.getElementById("chk-version").value;
  const policy = document.getElementById("chk-policy").value;
  if (!model || !version || !policy) { toast("All fields required", false); return; }

  const element = document.getElementById("slo-result");
  try {
    const data = await postJson("/slo/check", {
      model_name: model,
      model_version: version,
      policy_name: policy,
    });
    const rows = (data.checks || []).map((check) => `
      <tr><td class="mono">${escapeHtml(check.constraint)}</td>
      <td class="mono">${escapeHtml(check.threshold)}</td>
      <td class="mono">${escapeHtml(check.actual ?? "N/A")}</td>
      <td><span class="badge ${check.passed ? "badge-pass" : "badge-fail"}">${check.passed ? "PASS" : "FAIL"}</span></td></tr>`).join("");
    element.innerHTML = `<div class="tbl-wrap" style="margin-top:12px"><div class="tbl-title">Result: <span class="badge ${data.passed ? "badge-pass" : "badge-fail"}">${data.passed ? "PASS" : "FAIL"}</span></div><table><thead><tr><th>Constraint</th><th>Threshold</th><th>Actual</th><th>Status</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  } catch (error) {
    element.innerHTML = `<p style="color:var(--red);margin-top:12px">${escapeHtml(error.message)}</p>`;
  }
}

async function compareLlm() {
  const models = document.getElementById("llm-models").value.split(",").map((value) => value.trim()).filter(Boolean);
  const task = document.getElementById("llm-task").value || "general";
  const prompt = document.getElementById("llm-prompt").value;
  const jsonMode = document.getElementById("llm-json").value === "true";
  const element = document.getElementById("llm-result");
  if (!models.length || !prompt.trim()) { toast("Models and prompt required", false); return; }

  element.innerHTML = '<p style="color:var(--text2);font-size:13px;margin-top:12px">Comparing models...</p>';
  try {
    const data = await postJson("/llm/compare", { models, prompt, task, json_mode: jsonMode });
    const rows = (data.results || []).map((row) => {
      const bias = Number(row.bias_risk_score || 0);
      return `<tr><td class="mono">${escapeHtml(row.model_id)}</td><td>${escapeHtml(row.provider || "-")}</td><td class="mono">${Number(row.quality_score || 0).toFixed(3)}</td><td><span class="badge ${bias <= 0.2 ? "badge-pass" : "badge-fail"}">${bias.toFixed(3)}</span></td><td class="mono">${Number(row.latency_ms || 0).toFixed(2)} ms</td><td class="mono">$${Number(row.estimated_cost_usd || 0).toFixed(6)}</td><td><span class="badge ${row.live ? "badge-pass" : "badge-staging"}">${row.live ? "live" : "mock"}</span></td></tr>`;
    }).join("");
    element.innerHTML = `<div class="tbl-wrap" style="margin-top:12px;margin-bottom:0"><div class="tbl-title">Recommended: <span class="mono">${escapeHtml(data.recommended_model || "-")}</span></div><table><thead><tr><th>Model</th><th>Provider</th><th>Quality</th><th>Bias Risk</th><th>Latency</th><th>Cost</th><th>Mode</th></tr></thead><tbody>${rows}</tbody></table><div style="padding:12px 14px;color:var(--text2);font-size:12px">${escapeHtml(data.recommendation_reason || "")}</div></div>`;
    toast("LLM comparison complete", true);
  } catch (error) {
    element.innerHTML = `<p style="color:var(--red);margin-top:12px">${escapeHtml(error.message)}</p>`;
    toast(error.message, false);
  }
}

function renderShadowCharts() {
  const chartData = document.getElementById("shadow-chart-data");
  if (!chartData || typeof Chart === "undefined") return;
  const agreeData = JSON.parse(chartData.dataset.agreement);
  const latencyData = JSON.parse(chartData.dataset.latency);
  new Chart(document.getElementById("agreeChart"), {
    type: "doughnut",
    data: { labels: ["Agreed", "Disagreed"], datasets: [{ data: [agreeData.agreed, agreeData.disagreed], backgroundColor: ["#16a34a", "#dc2626"], borderWidth: 0 }] },
    options: { responsive: true, plugins: { legend: { labels: { color: "#5f6b7a" } } } },
  });
  new Chart(document.getElementById("latencyChart"), {
    type: "bar",
    data: { labels: latencyData.labels, datasets: [{ label: "Prod (ms)", data: latencyData.prod, backgroundColor: "#2563eb" }, { label: "Shadow (ms)", data: latencyData.shadow, backgroundColor: "#0f766e" }] },
    options: { responsive: true, scales: { x: { ticks: { color: "#5f6b7a" } }, y: { ticks: { color: "#5f6b7a" }, grid: { color: "#e5e7eb" } } }, plugins: { legend: { labels: { color: "#5f6b7a" } } } },
  });
}

document.addEventListener("DOMContentLoaded", renderShadowCharts);
