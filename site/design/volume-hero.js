// SPDX-License-Identifier: Apache-2.0
// The family's hero: a volume as three orthogonal slice planes drifting inside a wire cube, and a patch sweeping
// the axial plane region by region, the way the engine reads a volume. Drawn on a <canvas> behind a headline.
// No dependency. An ES module:
//
//   <canvas id="volume" aria-hidden="true"></canvas>
//   <script type="module">
//     import { mountVolumeHero } from "./design/volume-hero.js";
//     mountVolumeHero(document.getElementById("volume"));
//   </script>
//
// The canvas fills its positioned parent (CSS: position absolute, inset 0, width and height 100 %). Options, all
// optional: planes, three [solid, ink] hex pairs for the axial, sagittal and coronal planes; sweep, one pair for
// the patch; ground, the hex of the page below (the bottom of the canvas fades into it); wire, the hex of the cube
// and the grids; fps (30); grid, lines per plane (8); scale, the cube's size as a fraction of the shorter side
// (0.42); centre, [x, y] as fractions of the canvas ([0.5, 0.36]). The defaults read the kit's tokens from the
// canvas (--k-sage, --k-steel, --k-violet, --k-night). prefers-reduced-motion draws one still frame; a hidden tab
// stops the loop and a visible one restarts it. Returns { stop }.

export function mountVolumeHero(canvas, options) {
  if (!canvas || !canvas.getContext) return { stop() {} };
  const o = options || {};
  const css = getComputedStyle(canvas);
  const token = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
  const planes = (
    o.planes || [
      [token("--k-sage", "#52b89b"), token("--k-sage-ink", "#8fe0c6")],
      [token("--k-steel", "#5d9bf0"), token("--k-steel-ink", "#aacbff")],
      [token("--k-violet", "#8b7cf6"), token("--k-violet-ink", "#c4bbff")],
    ]
  ).map((pair) => pair.map(rgb));
  const sweep = (o.sweep || [token("--k-sage", "#52b89b"), token("--k-sage-ink", "#8fe0c6")]).map(rgb);
  const ground = rgb(o.ground || token("--k-night", "#080b12"));
  const wire = rgb(o.wire || "#b0c6e8");
  const minFrame = 1000 / (o.fps || 30);
  const lines = o.grid || 8;
  const scale = o.scale || 0.42;
  const centre = o.centre || [0.5, 0.36];

  const ctx = canvas.getContext("2d");
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const t0 = performance.now();
  let W = 0, H = 0, raf = 0, last = 0;

  function size() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const r = canvas.getBoundingClientRect();
    W = Math.max(1, Math.round(r.width));
    H = Math.max(1, Math.round(r.height));
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  // isometric projection of a unit cube
  function P(x, y, z, s, cx, cy) {
    return [cx + (x - z) * 0.86 * s, cy + (x + z) * 0.5 * s - y * s];
  }
  function quad(pts, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.stroke();
  }
  function grid(a, b, c, d, stroke) {
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 1; i < lines; i++) {
      const f = i / lines;
      ctx.moveTo(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f);
      ctx.lineTo(d[0] + (c[0] - d[0]) * f, d[1] + (c[1] - d[1]) * f);
      ctx.moveTo(a[0] + (d[0] - a[0]) * f, a[1] + (d[1] - a[1]) * f);
      ctx.lineTo(b[0] + (c[0] - b[0]) * f, b[1] + (c[1] - b[1]) * f);
    }
    ctx.stroke();
  }
  function frame(now) {
    raf = 0;
    if (!reduced) {
      if (now - last < minFrame) {
        raf = requestAnimationFrame(frame);
        return;
      }
      last = now;
    }
    const t = reduced ? 0 : (now - t0) / 1000;
    ctx.clearRect(0, 0, W, H);
    const s = Math.min(W, H) * scale, cx = W * centre[0], cy = H * centre[1];
    const sx = 0.5 + 0.28 * Math.sin(t * 0.21), sy = 0.5 + 0.28 * Math.sin(t * 0.17 + 1.7), sz = 0.5 + 0.28 * Math.sin(t * 0.13 + 3.1);
    const faint = rgba(wire, 0.08), edge = rgba(wire, 0.18);
    // the cube
    const c = [-0.5, 0.5].flatMap((y) => [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]].map(([x, z]) => P(x, y, z, s, cx, cy)));
    ctx.strokeStyle = edge;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (const [i, j] of [[0, 1], [1, 2], [2, 3], [3, 0], [4, 5], [5, 6], [6, 7], [7, 4], [0, 4], [1, 5], [2, 6], [3, 7]]) {
      ctx.moveTo(c[i][0], c[i][1]);
      ctx.lineTo(c[j][0], c[j][1]);
    }
    ctx.stroke();
    // the three planes, painter's order by depth: axial, sagittal, coronal
    const y = sy - 0.5, x = sx - 0.5, z = sz - 0.5;
    const slices = [
      { d: y, pts: [P(-0.5, y, -0.5, s, cx, cy), P(0.5, y, -0.5, s, cx, cy), P(0.5, y, 0.5, s, cx, cy), P(-0.5, y, 0.5, s, cx, cy)], hue: planes[0], a: 0.11, b: 0.35 },
      { d: x, pts: [P(x, -0.5, -0.5, s, cx, cy), P(x, -0.5, 0.5, s, cx, cy), P(x, 0.5, 0.5, s, cx, cy), P(x, 0.5, -0.5, s, cx, cy)], hue: planes[1], a: 0.09, b: 0.3 },
      { d: z, pts: [P(-0.5, -0.5, z, s, cx, cy), P(0.5, -0.5, z, s, cx, cy), P(0.5, 0.5, z, s, cx, cy), P(-0.5, 0.5, z, s, cx, cy)], hue: planes[2], a: 0.09, b: 0.3 },
    ].sort((p, q) => p.d - q.d);
    for (const p of slices) {
      quad(p.pts, rgba(p.hue[0], p.a), rgba(p.hue[1], p.b));
      grid(p.pts[0], p.pts[1], p.pts[2], p.pts[3], faint);
    }
    // the patch sweeping the axial plane: what the engine reads, region by region
    const k = (t * 0.12) % 1, px = -0.5 + ((k * 4) % 1) * 0.75, pz = -0.5 + Math.floor(k * 4) * 0.25;
    quad(
      [P(px, y, pz, s, cx, cy), P(px + 0.25, y, pz, s, cx, cy), P(px + 0.25, y, pz + 0.25, s, cx, cy), P(px, y, pz + 0.25, s, cx, cy)],
      rgba(sweep[0], 0.28),
      rgba(sweep[1], 0.7),
    );
    // the bottom fades into the page
    const f = ctx.createLinearGradient(0, H * 0.6, 0, H);
    f.addColorStop(0, rgba(ground, 0));
    f.addColorStop(1, rgba(ground, 1));
    ctx.fillStyle = f;
    ctx.fillRect(0, 0, W, H);
    if (!reduced && !document.hidden) raf = requestAnimationFrame(frame);
  }
  const start = () => { if (!raf) raf = requestAnimationFrame(frame); };
  const onResize = () => { size(); if (reduced) frame(performance.now()); };
  const onVisible = () => { if (!document.hidden) start(); };
  size();
  start();
  addEventListener("resize", onResize);
  document.addEventListener("visibilitychange", onVisible);
  return {
    stop() {
      if (raf) cancelAnimationFrame(raf);
      raf = 0;
      removeEventListener("resize", onResize);
      document.removeEventListener("visibilitychange", onVisible);
    },
  };
}

// "#rrggbb" to [r, g, b]; any other colour string is handed to the canvas as it is
function rgb(colour) {
  const m = /^#?([0-9a-f]{6})$/i.exec(String(colour).trim());
  if (!m) return colour;
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function rgba(c, a) {
  return Array.isArray(c) ? `rgba(${c[0]},${c[1]},${c[2]},${a})` : c;
}
