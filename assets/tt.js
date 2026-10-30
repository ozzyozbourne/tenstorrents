/* Tenstorrent From Zero: shared helpers for chapters and interactive figures. */
(function () {
  "use strict";

  // Copy buttons on every <pre>.
  function addCopyButtons() {
    document.querySelectorAll("pre").forEach(function (pre) {
      if (pre.querySelector(".copy")) return;
      var b = document.createElement("button");
      b.className = "copy";
      b.textContent = "copy";
      b.addEventListener("click", function () {
        var code = pre.querySelector("code") || pre;
        navigator.clipboard.writeText(code.innerText).then(function () {
          b.textContent = "copied";
          setTimeout(function () { b.textContent = "copy"; }, 1200);
        });
      });
      pre.appendChild(b);
    });
  }

  // Theme colors, mirrored from style.css for use in canvas drawing.
  var C = {
    bg: "#000000", bg2: "#161b22", bg3: "#1f2630", fg: "#e6e6e6", muted: "#9aa4b2",
    blue: "#58c4dd", yellow: "#ffd866", green: "#83c167", red: "#fc6255",
    purple: "#b48ead", orange: "#f0a35e", teal: "#5cd0b3", pink: "#ff79c6", border: "#2d3540"
  };

  // Create a HiDPI canvas inside a .viz container. Returns {canvas, ctx, W, H}.
  // Drawing uses logical units W x H regardless of device pixel ratio.
  function canvas(container, W, H) {
    var cv = document.createElement("canvas");
    var dpr = window.devicePixelRatio || 1;
    cv.width = W * dpr;
    cv.height = H * dpr;
    cv.style.aspectRatio = W + " / " + H;
    container.insertBefore(cv, container.firstChild);
    var ctx = cv.getContext("2d");
    ctx.scale(dpr, dpr);
    return { canvas: cv, ctx: ctx, W: W, H: H };
  }

  // Easing helpers in the spirit of manim's "smooth".
  function smooth(t) { t = Math.max(0, Math.min(1, t)); return t * t * (3 - 2 * t); }
  function lerp(a, b, t) { return a + (b - a) * t; }

  // A tiny step-based animation driver: steps is an array of
  // {dur: seconds, text: narration, draw: function(ctx, t in [0,1])}.
  // Provides play/pause/step/restart controls and narration.
  function stepper(opts) {
    var root = opts.root, steps = opts.steps, draw = opts.draw;
    var ctrl = document.createElement("div");
    ctrl.className = "controls";
    var narr = document.createElement("div");
    narr.className = "narration";
    var bPlay = btn("Play"), bPrev = btn("Back"), bNext = btn("Step"), bReset = btn("Restart");
    var info = document.createElement("span");
    info.style.color = C.muted;
    info.style.marginLeft = "auto";
    [bPrev, bPlay, bNext, bReset, info].forEach(function (e) { ctrl.appendChild(e); });
    root.appendChild(ctrl);
    root.appendChild(narr);

    var i = 0, t = 0, playing = false, last = 0;
    function btn(label) { var b = document.createElement("button"); b.textContent = label; return b; }
    function render() {
      draw(i, smooth(t), steps[i]);
      narr.innerHTML = steps[i].text || "";
      info.textContent = "step " + (i + 1) + " / " + steps.length;
      bPlay.textContent = playing ? "Pause" : "Play";
    }
    function frame(now) {
      if (!playing) return;
      var dt = (now - last) / 1000;
      last = now;
      t += dt / (steps[i].dur || 1.5);
      if (t >= 1) {
        t = 1;
        if (i < steps.length - 1) { render(); i++; t = 0; }
        else { playing = false; }
      }
      render();
      if (playing) requestAnimationFrame(frame);
    }
    bPlay.onclick = function () {
      if (!playing && i === steps.length - 1 && t >= 1) { i = 0; t = 0; }
      playing = !playing;
      last = performance.now();
      render();
      if (playing) requestAnimationFrame(frame);
    };
    bNext.onclick = function () { playing = false; if (t < 1) t = 1; else if (i < steps.length - 1) { i++; t = 1; } render(); };
    bPrev.onclick = function () { playing = false; if (i > 0) i--; t = 1; render(); };
    bReset.onclick = function () { playing = false; i = 0; t = 0; render(); };
    render();
    return { render: render };
  }

  // Rounded rectangle with optional fill/stroke and centered label.
  function box(ctx, x, y, w, h, o) {
    o = o || {};
    var r = o.r == null ? 6 : o.r;
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
    if (o.fill) { ctx.globalAlpha = o.alpha == null ? 1 : o.alpha; ctx.fillStyle = o.fill; ctx.fill(); ctx.globalAlpha = 1; }
    if (o.stroke) { ctx.strokeStyle = o.stroke; ctx.lineWidth = o.lw || 1.5; ctx.stroke(); }
    if (o.label) {
      ctx.fillStyle = o.color || C.fg;
      ctx.font = (o.font || "13px") + " " + (o.mono ? "ui-monospace, monospace" : "Inter, sans-serif");
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(o.label, x + w / 2, y + h / 2);
    }
  }

  function text(ctx, s, x, y, o) {
    o = o || {};
    ctx.fillStyle = o.color || C.fg;
    ctx.font = (o.font || "13px") + " " + (o.mono ? "ui-monospace, monospace" : "Inter, sans-serif");
    ctx.textAlign = o.align || "left";
    ctx.textBaseline = o.base || "middle";
    ctx.fillText(s, x, y);
  }

  function arrow(ctx, x1, y1, x2, y2, o) {
    o = o || {};
    ctx.strokeStyle = o.color || C.fg;
    ctx.fillStyle = o.color || C.fg;
    ctx.lineWidth = o.lw || 2;
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    var a = Math.atan2(y2 - y1, x2 - x1), s = o.head || 8;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - s * Math.cos(a - 0.4), y2 - s * Math.sin(a - 0.4));
    ctx.lineTo(x2 - s * Math.cos(a + 0.4), y2 - s * Math.sin(a + 0.4));
    ctx.closePath(); ctx.fill();
  }

  function clear(ctx, W, H) { ctx.fillStyle = C.bg; ctx.fillRect(0, 0, W, H); }

  window.TT = { C: C, canvas: canvas, smooth: smooth, lerp: lerp, stepper: stepper, box: box, text: text, arrow: arrow, clear: clear };
  document.addEventListener("DOMContentLoaded", addCopyButtons);
})();
