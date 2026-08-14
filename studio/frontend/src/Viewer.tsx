// SPDX-License-Identifier: Apache-2.0

import { useEffect, useRef, useState } from "react";
import { Niivue } from "@niivue/niivue";
import { postJson } from "./api";

// The 3D Slicer mark, inlined so the button needs no asset and no network. 3D Slicer is a trademark
// of its project; shown here to name the application this hands the volume over to.
const SLICER_MARK = "data:image/webp;base64,UklGRv4EAABXRUJQVlA4WAoAAAAQAAAAIwAAIwAAQUxQSIABAAABkGttu2k735xz7xOjTLpkd7Ft25VtdE5uICM3wNpJxc62bbNzMvF/xeLMFUTEBCDTAHrQlnOfgz3Rt03bNu1qKK40mqy8JkyVv//+2PtVqAIamP2CpA9CBiaf1xUxaL6PdIGZIl6eFNFofZcuMHfg0/raaK0yNFq9pGPBwCfIrXTTu3Qs/vv0qVMnT/eCThjsoWPpk2EAGEyjY6nee+vHpagGLySUQzJwUqVepWKwgI4RBiOpL4svT7htxarlq0d0dYx/fDN9FO/cX7f/OEMUkp4HX1DiHfrJ6J77GD/wvMQj7Y//QZ5RYgm/HWGIFXhxPX0sz60dLWOLDFLnxMcJcreKmYzkuRqq3iMJMYK8b6oMxtLFsJwJA4NdtOU5HoIBlGlwma4sxwfNtQKg0eI+bTmWr2rQSGq0vEgfigXHuzUYpGs03EF6L7mCI3c3h0G2BibeJumDJCR4ko+nAxp5lUF19sm/zGnPL2oIrVDQAOi05vCT78Kfz45v6g7AIBNWUDggWAMAAPARAJ0BKiQAJAA+PRyLRCIhoROVJCADxLMAU6PjxHfbuGLeq7UeYDHL/QA8p72Rv3Iron4udaD4W9aMmu99rgdED+17zv/AehNp2P1V9e7/C8xPy//wf7b8BX8k/oH93/N3+3fKr1G36zFEj+CfKJ4Srmqwk6efhqGKbAKWuBuXMlU+Kw2+k55WSRJl5Ex6f5FswKah1lDgAP75yFUGAPoQZSsy4D/gwaO/74yWMIJ3uzU8Q99Nz3zV7UndAmPOkqTMGih8sZtqVbrkfXbsmtdoIDj9/yhx3zHPGp739nZPS5TfB/KpH1RdLDs9azs2Wk4vwlHCq0bUnSiVqk6tvrRCcxYlNe/99aE1bif54UCm4EPKbRLMPNWf9zSAiUC1P05443NzFrvZnuNgzxR8+AFk9bHGcVy7n3WRTuEP/eznu+tUI0mpr/ct45N56wMnxe3C+f+SSIH+SEaQmZDLCV2ICm3f4PlF0VeSgPx6StoIngqvyHzXMv+a1/5bI7MB40QLbkjzVry9NoFzEh9fvOvMzqzkEJj9/YGJkJ6UMMkY+6ZQH/IC+9WB+J2r4f9cf8Uo96lPAE+l1W3aty63k0gPwRUKIX+aCGhM8O+fitNlCjp8URdcXzrrIP2z8zVZiMr3nBzqVaw54GvznqnUrJ+/+h1qbwTC1WkmxEqvapBtnXGy+enAvuh36gev51I1Ye9x+XVv7h3dO9a+eT/6COImcvDYbEeiYkaGick1PPEgq88QSm2S5gAbIg+a6w1eD1Ugeizdhy8y9nn8JSZ1CLCu+OXlcX+k2t0wVkwuqCHGb9E2gG0lWcUaAYmdjGAcNuMD/pOxE8DhXIKZn2Ceei5DdzbSNHZogzR/38WEWoHj+Hz7ras/u7q/nSsvXgztRvZzFcBVYv2souHMa/y+0CY57aHxi94/+QPH+PMIr5javcKnBCsSFgYi0gsHr2sI0a755CsCU0x+BHLEp9C+RbmD671ozy3NJUOAQg+wBswdGYno0/3AWPPwrnxnis6JgEQr0B9lHJHXeXLTL/7c5NON2//6COQ5uw8FMTaiZE14LWNqVpCdtlAj/fWd9DXh5p+9X2Ydq9+sC0yeXALAM30FwHTWrQviKngLyGt/nxe0N/860Q4AAAA=";

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
  const [failed, setFailed] = useState("");

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
    setFailed("");
    nv.loadVolumes([{ url: `/files/volume?path=${encodeURIComponent(path)}`, name }])
      .then(() => onDims?.(dimsLabel(nv)))
      .catch(() => {
        // Leaving the previous volume on screen under a name that failed reads as a successful load.
        for (const loaded of [...nv.volumes]) nv.removeVolume(loaded);
        onDims?.("");
        setFailed(name);
      });
  }, [path, onDims]);

  return (
    <div className="canvas-wrap">
      <canvas ref={canvasRef} />
      {failed && (
        <div className="v-failed">
          <p>
            Could not open <b>{failed}</b>. The file may be missing, or in a format the viewer does not read
            (NIfTI, MHA/MHD, NRRD, DICOM).
          </p>
        </div>
      )}
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
  const [slicerNote, setSlicerNote] = useState("");
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

  // Hand the loaded volume(s) over to 3D Slicer: into the running instance when its Web Server module
  // listens, else by launching Slicer on them. Same machine, same paths: nothing is copied.
  async function openInSlicer() {
    const paths = [path, compare ? comparePath : null].filter(Boolean);
    if (paths.length === 0) return;
    setSlicerNote("…");
    try {
      const d = await postJson("/api/slicer/open", { paths });
      setSlicerNote(d.ok ? (d.via === "webserver" ? "sent to Slicer ✓" : "starting Slicer…") : d.detail || "failed");
    } catch {
      setSlicerNote("failed");
    }
    window.setTimeout(() => setSlicerNote(""), 6000);
  }

  // Escape leaves fullscreen: the whole app is covered while it's on.
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
          {slicerNote && <span className="v-note">{slicerNote}</span>}
          <button
            className="v-btn"
            onClick={openInSlicer}
            disabled={!path}
            title="Open the loaded volume(s) in 3D Slicer (the running instance if its Web Server listens, else a new one)"
          >
            <img className="v-mark" src={SLICER_MARK} alt="" />
            3D Slicer
          </button>
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
            placeholder="compare volume: or click one in the tree"
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
