// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";

// The loaded volume's voxel grid + spacing, e.g. "256×256×180 · 1.0×1.0×1.0 mm".
function dimsLabel(nv: Niivue | null): string {
  const img = nv?.volumes?.[0];
  if (!img?.hdr) return "";
  const d = img.hdr.dims;
  const p = img.hdr.pixDims;
  const grid = `${d[1]}×${d[2]}×${d[3]}`;
  const mm = p ? ` · ${p[1].toFixed(2)}×${p[2].toFixed(2)}×${p[3].toFixed(2)} mm` : "";
  return grid + mm;
}

// One NiiVue canvas that follows a path prop. Reused for the primary view and the compare pane; `onReady`
// hands the instance up so the two panes can be crosshair/zoom-coupled.
function Canvas({ path, onDims, onReady }: { path: string | null; onDims?: (d: string) => void; onReady?: (nv: Niivue) => void }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const nvRef = useRef<Niivue | null>(null);

  useEffect(() => {
    const nv = new Niivue({ backColor: [0.035, 0.08, 0.09, 1], show3Dcrosshair: true });
    nvRef.current = nv;
    nv.attachToCanvas(canvasRef.current!);
    onReady?.(nv);
    return () => {
      nvRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !path) return;
    const name = path.split("/").pop() || "volume.nii.gz";
    nv.loadVolumes([{ url: `/files/volume?path=${encodeURIComponent(path)}`, name }])
      .then(() => onDims?.(dimsLabel(nv)))
      .catch((e) => onDims?.("failed: " + String(e)));
  }, [path, onDims]);

  return (
    <div className="canvas-wrap">
      <canvas ref={canvasRef} />
    </div>
  );
}

export default function Viewer({
  path,
  onPathChange,
  comparePath,
  onComparePathChange,
  compareMode,
  onCompareModeChange,
}: {
  path: string | null;
  onPathChange: (p: string) => void;
  comparePath?: string | null;
  onComparePathChange?: (p: string) => void;
  compareMode?: boolean;
  onCompareModeChange?: (on: boolean) => void;
}) {
  const [field, setField] = useState("");
  const [field2, setField2] = useState("");
  const [dims, setDims] = useState("");
  const [dims2, setDims2] = useState("");
  const [full, setFull] = useState(false);
  const compare = !!compareMode; // explicit toggle: when on, a second pane opens and tree clicks fill it
  const nvA = useRef<Niivue | null>(null);
  const nvB = useRef<Niivue | null>(null);

  // Link the two panes so panning/scrolling/zooming one drives the other (crosshair + 3D view).
  const couple = () => {
    if (nvA.current && nvB.current) {
      nvA.current.broadcastTo(nvB.current, { "2d": true, "3d": true });
      nvB.current.broadcastTo(nvA.current, { "2d": true, "3d": true });
    }
  };

  useEffect(() => {
    if (path) setField(path);
  }, [path]);
  useEffect(() => {
    setField2(comparePath ?? "");
  }, [comparePath]);

  // Escape leaves fullscreen — the whole app is covered while it's on.
  useEffect(() => {
    if (!full) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setFull(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [full]);

  return (
    <section className={full ? "pane viewer fullscreen" : "pane viewer"}>
      <div className="pane-head">
        Viewer <span className="accent">· NiiVue</span>
        {dims && (
          <span className="v-dims" title={dims}>
            {dims}
          </span>
        )}
        <span className="v-tools">
          <button
            className={compare ? "v-btn on" : "v-btn"}
            onClick={() => onCompareModeChange?.(!compare)}
            title={compare ? "Back to a single view" : "Compare two volumes side by side"}
          >
            Compare
          </button>
          <button className="v-btn" onClick={() => setFull((f) => !f)} title={full ? "Exit fullscreen (Esc)" : "Enlarge"}>
            {full ? "Exit ⤢" : "Enlarge ⤢"}
          </button>
        </span>
      </div>
      <div className="vbar">
        <input
          value={field}
          onChange={(e) => setField(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && onPathChange(field)}
          placeholder="/path/to/volume.nii.gz or .mha"
        />
        <button onClick={() => onPathChange(field)}>Load</button>
      </div>
      {compare && (
        <div className="vbar">
          <input
            value={field2}
            onChange={(e) => setField2(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onComparePathChange?.(field2)}
            placeholder="compare volume — or click one in the tree"
          />
          <button onClick={() => onComparePathChange?.(field2)}>Load</button>
          {dims2 && <span className="v-dims">{dims2}</span>}
        </div>
      )}
      <div className={compare ? "v-split" : "v-single"}>
        <Canvas
          path={path}
          onDims={setDims}
          onReady={(nv) => {
            nvA.current = nv;
            couple();
          }}
        />
        {compare &&
          (comparePath ? (
            <Canvas
              path={comparePath}
              onDims={setDims2}
              onReady={(nv) => {
                nvB.current = nv;
                couple();
              }}
            />
          ) : (
            <div className="canvas-wrap v-empty">Click a volume in the tree to compare</div>
          ))}
      </div>
    </section>
  );
}
