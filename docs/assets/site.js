"use strict";

(() => {
  const data = JSON.parse(document.getElementById("experiment-data").textContent);
  const names = { miou: "mIoU", clip_aes: "CLIP-AES", clip_text: "CLIP-Text", clip_image: "CLIP-Image" };
  const precision = { miou: 2, clip_aes: 4, clip_text: 4, clip_image: 4 };

  const dialog = document.getElementById("image-dialog");
  const enlarged = document.getElementById("full-image");
  const caption = document.getElementById("image-caption");
  const originalLink = document.getElementById("original-image-link");
  document.querySelectorAll(".image-button").forEach(button => {
    button.addEventListener("click", () => {
      enlarged.src = button.dataset.full;
      enlarged.alt = button.dataset.caption;
      caption.textContent = button.dataset.caption;
      originalLink.href = button.dataset.full;
      dialog.showModal();
    });
  });
  document.getElementById("close-dialog").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", event => {
    if (event.target !== dialog) return;
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });

  function renderStyleChart() {
    const metric = document.getElementById("style-metric").value;
    const rows = data.styleComparison.filter(row => row.style !== "Overall");
    const maximum = Math.max(...rows.flatMap(row => [row.cloudnet[metric], row.freestylenet[metric]])) * 1.15;
    const host = document.getElementById("style-chart");
    host.replaceChildren();
    const grid = document.createElement("div");
    grid.className = "style-bars";
    rows.forEach(row => {
      const label = document.createElement("div");
      label.className = "bar-label";
      label.textContent = row.style;
      grid.append(label);
      const pair = document.createElement("div");
      pair.className = "bar-pair";
      ["cloudnet", "freestylenet"].forEach(method => {
        const score = row[method][metric];
        const track = document.createElement("div");
        track.className = "bar-track";
        track.setAttribute("aria-label", `${row.style}, ${method === "cloudnet" ? "CloudNet" : "FreestyleNet"}, ${names[metric]}: ${score}`);
        const fill = document.createElement("div");
        fill.className = `bar-fill${method === "freestylenet" ? " freestyle" : ""}`;
        fill.style.width = `${score / maximum * 100}%`;
        const value = document.createElement("span");
        value.className = "bar-value";
        value.textContent = score.toFixed(precision[metric]);
        track.append(fill, value);
        pair.append(track);
      });
      grid.append(pair);
    });
    host.append(grid);
  }

  function renderScaleChart(metric) {
    const NS = "http://www.w3.org/2000/svg";
    const make = (tag, attributes, content) => {
      const element = document.createElementNS(NS, tag);
      Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
      if (content !== undefined) element.textContent = content;
      return element;
    };
    const rows = data.scales;
    const values = rows.map(row => row[metric]);
    const low = Math.min(...values), high = Math.max(...values);
    const padding = Math.max((high - low) * .35, .0001);
    const bottom = low - padding, top = high + padding;
    const y = value => 178 - (value - bottom) / (top - bottom) * 134;
    const x = scale => 92 + (Number(scale) - .5) * 708;
    const svg = make("svg", { viewBox: "0 0 890 230", role: "img", "aria-label": `${names[metric]} across five style token scales. Vertical axis spans ${bottom.toFixed(precision[metric])} to ${top.toFixed(precision[metric])}.` });
    svg.append(make("title", {}, `${names[metric]} by style token scale`));
    svg.append(make("desc", {}, rows.map(row => `Scale ${row.name}: ${row[metric]}`).join("; ")));
    svg.append(make("text", { x: 15, y: 20, fill: "#1c3337", "font-size": 11, "font-family": "Arial, sans-serif" }, `${names[metric]} ↑`));
    for (let i = 0; i < 4; i++) {
      const value = bottom + (top - bottom) * i / 3;
      const yy = y(value);
      svg.append(make("line", { x1: 90, x2: 815, y1: yy, y2: yy, stroke: "#e2e7df", "stroke-width": 1 }));
      svg.append(make("text", { x: 73, y: yy + 4, "text-anchor": "end", fill: "#607170", "font-size": 10, "font-family": "Arial, sans-serif" }, value.toFixed(precision[metric])));
    }
    svg.append(make("polyline", { points: rows.map(row => `${x(row.name)},${y(row[metric])}`).join(" "), fill: "none", stroke: "#196a60", "stroke-width": 2 }));
    rows.forEach(row => {
      const xx = x(row.name), yy = y(row[metric]);
      if (row.name === "1.0") svg.append(make("circle", { cx: xx, cy: yy, r: 10, fill: "#196a6018" }));
      svg.append(make("circle", { cx: xx, cy: yy, r: 4, fill: "#196a60", stroke: "white", "stroke-width": 1.5 }));
      svg.append(make("text", { x: xx, y: yy - 13, "text-anchor": "middle", fill: "#1c3337", "font-size": 11, "font-family": "Arial, sans-serif" }, row[metric].toFixed(precision[metric])));
      svg.append(make("text", { x: xx, y: 204, "text-anchor": "middle", fill: "#607170", "font-size": 10, "font-family": "Arial, sans-serif" }, row.name === "1.0" ? "1.0 · main" : row.name));
    });
    svg.append(make("text", { x: 815, y: 224, "text-anchor": "end", fill: "#607170", "font-size": 10, "font-family": "Arial, sans-serif" }, "Style token scale λ"));
    document.getElementById("scale-chart").replaceChildren(svg);
  }

  document.getElementById("style-metric").addEventListener("change", renderStyleChart);
  document.querySelectorAll("[data-trend]").forEach(button => {
    button.addEventListener("click", () => {
      document.querySelectorAll("[data-trend]").forEach(tab => tab.setAttribute("aria-pressed", String(tab === button)));
      renderScaleChart(button.dataset.trend);
    });
  });
  renderStyleChart();
  renderScaleChart("clip_aes");
})();
