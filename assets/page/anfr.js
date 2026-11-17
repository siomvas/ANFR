/* ANFR page: the Table 2 explorer and the Table 6 privacy chart. Every number is from the paper. */
(function () {
  "use strict";
  var REDUCED = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var NF = new Intl.NumberFormat("en", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  var ARCH = ["BN-ResNet", "GN-ResNet", "SE-ResNet", "NF-ResNet", "ANFR"];
  var T2 = {
    "Fed-ISIC2019": { m: "balanced accuracy", lo: 60, hi: 80, r: {
      "FedAvg": [[66.01, .73], [65.09, .42], [65.29, 1.32], [72.49, .60], [74.78, .16]],
      "FedProx": [[66.49, .41], [66.51, 1.21], [66.29, .63], [71.28, 2.14], [75.61, .71]],
      "FedAdam": [[65.88, .67], [64.60, .39], [65.18, 1.90], [69.96, .14], [73.02, .93]],
      "SCAFFOLD": [[65.41, .72], [68.84, .46], [68.99, .18], [73.30, .50], [76.52, .60]] } },
    "FedChest": { m: "mean AUROC", lo: 80, hi: 85, r: {
      "FedAvg": [[82.80, .13], [83.40, .25], [82.14, .18], [83.40, .11], [83.49, .14]],
      "FedProx": [[82.14, .10], [82.04, .08], [81.50, .26], [81.26, .58], [82.14, .10]],
      "FedAdam": [[83.02, .11], [82.11, .10], [82.72, .16], [83.10, .09], [83.33, .07]],
      "SCAFFOLD": [[83.52, .14], [83.95, .05], [83.50, .08], [84.06, .02], [84.26, .10]] } },
    "CIFAR-10": { m: "accuracy", lo: 90, hi: 98, r: {
      "FedAvg": [[91.71, .74], [96.60, .11], [94.07, .04], [96.72, .05], [97.42, .01]],
      "FedProx": [[95.03, .04], [96.05, .04], [94.60, .07], [96.82, .04], [96.33, .09]],
      "FedAdam": [[91.23, .29], [95.80, .24], [94.09, .17], [95.54, .10], [96.93, .06]],
      "SCAFFOLD": [[92.51, .99], [96.78, .01], [94.30, .03], [96.84, .01], [97.38, .03]] } }
  };
  function say(t) { var s = document.getElementById("anfr-status"); if (s) { s.textContent = ""; setTimeout(function () { s.textContent = t; }, 30); } }
  function el(tag, a, p) { var n = document.createElementNS("http://www.w3.org/2000/svg", tag); for (var k in a) n.setAttribute(k, a[k]); if (p) p.appendChild(n); return n; }
  function rank(v) {
    var a = v[4][0], best = Math.max.apply(null, v.slice(0, 4).map(function (x) { return x[0]; }));
    return a > best ? "Best" : a === best ? "Joint best" : "Second";
  }

  /* ---- Table 2 explorer ---- */
  var list = document.getElementById("t2-rows"), open = document.getElementById("t2-open");
  if (list && open) {
    var rows = [];
    Object.keys(T2).forEach(function (ds) { Object.keys(T2[ds].r).forEach(function (m) { rows.push([ds, m]); }); });
    list.innerHTML = rows.map(function (r, i) {
      var v = T2[r[0]].r[r[1]], k = rank(v);
      return '<li><button class="t2-row" type="button" aria-pressed="' + (i === 0) + '" data-i="' + i + '"><span>' + r[0] + '</span><span>' + r[1] + '</span><span class="t2-row__v">' + NF.format(v[4][0]) + '</span><span class="t2-chip' + (k !== "Second" ? " t2-chip--best" : "") + '">' + k + '</span></button></li>';
    }).join("");
    var cs = getComputedStyle(open);
    function draw(i, animate) {
      var ds = rows[i][0], m = rows[i][1], d = T2[ds], v = d.r[m];
      document.getElementById("t2-title").textContent = ds + ", " + m;
      document.getElementById("t2-metric").textContent = d.m + ", mean ± std";
      var W = 640, L = 112, R = 72, rowH = 64, top = 18, H = top + 5 * rowH + 34;
      var ink = cs.getPropertyValue("--u-ink").trim() || "#f1f1ef", mut = cs.getPropertyValue("--u-muted").trim() || "#9a9b96", line = cs.getPropertyValue("--u-line").trim() || "rgba(255,255,255,.1)";
      var acc = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim() || "#8fa68e";
      function x(val) { return L + (val - d.lo) / (d.hi - d.lo) * (W - L - R); }
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": ds + " with " + m + ", " + d.m + ": " + ARCH.map(function (a, j) { return a + " " + NF.format(v[j][0]); }).join(", ") });
      for (var t = d.lo; t <= d.hi; t += (d.hi - d.lo) / 5) {
        el("line", { x1: x(t), x2: x(t), y1: top - 4, y2: H - 28, stroke: line }, svg);
        el("text", { x: x(t), y: H - 8, "text-anchor": "middle", fill: mut, "font-size": 12 }, svg).textContent = Math.round(t);
      }
      ARCH.forEach(function (a, j) {
        var cy = top + j * rowH + rowH / 2, me = j === 4, c = me ? acc : ink;
        el("text", { x: 0, y: cy + 5, fill: me ? ink : mut, "font-size": 14, "font-weight": me ? 500 : 400 }, svg).textContent = a;
        var g = el("g", {}, svg), lo = x(v[j][0] - v[j][1]), hi = x(v[j][0] + v[j][1]), cx = x(v[j][0]);
        el("line", { x1: x(d.lo), x2: cx, y1: cy, y2: cy, stroke: me ? acc : line, "stroke-width": me ? 3 : 2, "stroke-linecap": "round" }, g);
        el("line", { x1: lo, x2: hi, y1: cy, y2: cy, stroke: c, "stroke-width": 1.5 }, g);
        el("line", { x1: lo, x2: lo, y1: cy - 6, y2: cy + 6, stroke: c, "stroke-width": 1.5 }, g);
        el("line", { x1: hi, x2: hi, y1: cy - 6, y2: cy + 6, stroke: c, "stroke-width": 1.5 }, g);
        el("circle", { cx: cx, cy: cy, r: me ? 7 : 5.5, fill: c }, g);
        el("text", { x: W - R + 12, y: cy + 5, fill: me ? acc : ink, "font-size": 14, "font-weight": me ? 500 : 400 }, svg).textContent = NF.format(v[j][0]);
        if (animate && !REDUCED) {
          g.style.transformBox = "fill-box"; g.style.transformOrigin = "left center";
          g.animate([{ transform: "scaleX(0.6)", opacity: 0 }, { transform: "scaleX(1)", opacity: 1 }], { duration: 420, delay: j * 50, easing: "cubic-bezier(.2,.7,.1,1)", fill: "backwards" });
        }
      });
      open.innerHTML = ""; open.appendChild(svg);
    }
    list.addEventListener("click", function (e) {
      var b = e.target.closest(".t2-row"); if (!b) return;
      list.querySelectorAll(".t2-row").forEach(function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
      var i = +b.dataset.i; draw(i, true);
      try { var u = new URL(location.href); u.searchParams.set("setting", rows[i][0] + "-" + rows[i][1]); history.replaceState(null, "", u); } catch (er) {}
      say(rows[i][0] + ", " + rows[i][1] + ": ANFR " + NF.format(T2[rows[i][0]].r[rows[i][1]][4][0]) + ", " + rank(T2[rows[i][0]].r[rows[i][1]]).toLowerCase() + ".");
    });
    var want = new URLSearchParams(location.search).get("setting"), first = 0;
    rows.forEach(function (r, i) { if (want === r[0] + "-" + r[1]) first = i; });
    list.querySelectorAll(".t2-row").forEach(function (o, i) { o.setAttribute("aria-pressed", i === first ? "true" : "false"); });
    draw(first, false);
  }

  /* ---- Table 6: FedPathology accuracy without and with DP-SGD at epsilon 1 ---- */
  var host = document.getElementById("dp-chart"), sw = document.getElementById("dp-switch");
  if (host && sw) {
    var DP = [["GN-ResNet", 84.79, 2.72, 67.27, 5.08], ["ANFR", 84.47, 3.08, 81.11, 0.33]], on = false, sec = document.getElementById("privacy");
    function render(animate) {
      var cs2 = getComputedStyle(sec), ink = cs2.getPropertyValue("--ink").trim() || "#f1f1ef", mut = cs2.getPropertyValue("--muted").trim() || "#9a9b96", line = cs2.getPropertyValue("--line").trim() || "rgba(255,255,255,.12)", acc = cs2.getPropertyValue("--accent").trim() || "#8fa68e";
      var W = Math.max(320, host.clientWidth || 1200), narrow = W < 640, L = narrow ? 0 : 200, R = 40, rowH = narrow ? 160 : 150, top = 10, H = top + 2 * rowH + 40, lo = 60, hi = 90;
      function x(v) { return L + (v - lo) / (hi - lo) * (W - L - R); }
      var svg = el("svg", { viewBox: "0 0 " + W + " " + H, width: W, height: H, role: "img", "aria-label": on ? "With DP-SGD at epsilon 1: ANFR 81.11, down from 84.47; GN-ResNet 67.27, down from 84.79." : "Without differential privacy: ANFR 84.47, GN-ResNet 84.79." });
      for (var t = lo; t <= hi; t += 5) {
        el("line", { x1: x(t), x2: x(t), y1: top, y2: H - 30, stroke: line }, svg);
        el("text", { x: x(t), y: H - 8, "text-anchor": "middle", fill: mut, "font-size": 14 }, svg).textContent = t + "%";
      }
      DP.forEach(function (d, j) {
        var cy = top + j * rowH + (narrow ? 78 : rowH / 2), me = d[0] === "ANFR", c = me ? acc : ink;
        if (narrow) el("text", { x: 0, y: cy - 36, fill: ink, "font-size": 17, "font-weight": 500 }, svg).textContent = d[0];
        else el("text", { x: 0, y: cy + 6, fill: ink, "font-size": 19, "font-weight": 500 }, svg).textContent = d[0];
        var from = x(d[1]), to = x(on ? d[3] : d[1]);
        var ghost = el("circle", { cx: from, cy: cy, r: 9, fill: "none", stroke: mut, "stroke-width": 1.5, opacity: on ? 1 : 0 }, svg);
        var bar = el("line", { x1: from, x2: from, y1: cy, y2: cy, stroke: c, "stroke-width": 3, "stroke-linecap": "round", opacity: .55 }, svg);
        var dot = el("circle", { cx: from, cy: cy, r: 11, fill: c }, svg);
        var lab = el("text", { x: from, y: cy - 22, "text-anchor": "middle", fill: c, "font-size": 18, "font-weight": 500 }, svg);
        function set(e) {
          var cx = from + (to - from) * e; dot.setAttribute("cx", cx); bar.setAttribute("x2", cx);
          lab.setAttribute("x", cx); lab.textContent = on ? NF.format(d[3]) + "  (−" + NF.format(d[1] - d[3]) + ")" : NF.format(d[1]);
        }
        if (animate && !REDUCED) { var t0 = null; requestAnimationFrame(function f(ts) { if (t0 === null) t0 = ts; var p = Math.min(1, (ts - t0) / 700), e = 1 - Math.pow(1 - p, 3); set(e); if (p < 1) requestAnimationFrame(f); }); }
        else set(1);
      });
      host.innerHTML = ""; host.appendChild(svg);
    }
    sw.addEventListener("click", function () {
      on = !on; sw.setAttribute("aria-checked", on ? "true" : "false"); render(true);
      say(on ? "With DP-SGD at epsilon 1, ANFR drops 3.36 points and GN-ResNet 17.52." : "Without differential privacy.");
    });
    render(false);
    if ("ResizeObserver" in window) { var lw = 0; new ResizeObserver(function () { var w = host.clientWidth; if (Math.abs(w - lw) > 4) { lw = w; render(false); } }).observe(host); }
  }

  document.querySelectorAll("[data-copy]").forEach(function (b) {
    b.addEventListener("click", function () {
      var src = document.getElementById(b.getAttribute("data-copy")), label = b.textContent;
      if (navigator.clipboard && window.isSecureContext) navigator.clipboard.writeText(src ? src.textContent : "").then(function () { b.textContent = "Copied"; say("BibTeX copied."); setTimeout(function () { b.textContent = label; }, 1400); }, function () {});
    });
  });
})();
