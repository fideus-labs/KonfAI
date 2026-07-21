// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

// Navigate the compute node's filesystem to pick a folder (or, with onPickFile, a file).
// Data/outputs never leave the machine — the browser only points Studio at what already lives here.
export default function FolderBrowser({
  start,
  onPick,
  onPickFile,
  onClose,
  title = "Choose a folder",
  cta = "Use this folder",
}: {
  start: string;
  onPick?: (path: string) => void;
  onPickFile?: (path: string) => void;
  onClose: () => void;
  title?: string;
  cta?: string;
}) {
  const [path, setPath] = useState(start);
  const [resolved, setResolved] = useState("");
  const [dirs, setDirs] = useState<string[]>([]);
  const [files, setFiles] = useState<string[]>([]);
  const [parent, setParent] = useState<string | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    const q = path ? `?path=${encodeURIComponent(path)}` : "";
    fetch(`/api/browse${q}`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`${r.status} ${r.statusText}`))))
      .then((d) => {
        setResolved(d.path);
        setDirs(d.dirs);
        setFiles(d.files ?? []);
        setParent(d.parent);
        setError("");
      })
      .catch((e) => setError(`Can't open this folder (${(e as Error).message}).`));
  }, [path]);

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">{title}</div>
        <div className="modal-path">{resolved || "…"}</div>
        {error && <div className="modal-err">{error}</div>}
        <div className="modal-list">
          {parent && (
            <button className="dir up" onClick={() => setPath(parent)}>
              ‹ ..
            </button>
          )}
          {dirs.map((d) => (
            <button key={d} className="dir" onClick={() => setPath(`${resolved}/${d}`)}>
              ▸ {d}
            </button>
          ))}
          {onPickFile &&
            files.map((f) => (
              <button key={f} className="file" onClick={() => onPickFile(`${resolved}/${f}`)}>
                <span className="fdot" /> {f}
              </button>
            ))}
          {dirs.length === 0 && (!onPickFile || files.length === 0) && (
            <div className="modal-empty">nothing here</div>
          )}
        </div>
        <div className="modal-foot">
          <button className="ghost" onClick={onClose}>
            Cancel
          </button>
          {onPick && (
            <button className="primary" onClick={() => onPick(resolved)} disabled={!resolved}>
              {cta}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
