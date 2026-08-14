// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";

// Navigate the compute node's filesystem to pick a folder (or, with onPickFile, a file).
// Data/outputs never leave the machine: the browser only points Studio at what already lives here.
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
  // The path, editable. Clicking through the tree from $HOME to a mounted share is a long walk, and a
  // path is the one thing a user already has: they can paste it. Navigation keeps this in step.
  const [typed, setTyped] = useState(start);
  const [reload, setReload] = useState(0); // Enter on the path always re-reads, even for the same path

  useEffect(() => {
    // The previous listing is invalidated before the request: keeping it would let the primary
    // action submit the old folder while the input shows an invalid path. The cleanup aborts a
    // slower earlier request so it cannot overwrite a newer navigation's result.
    const controller = new AbortController();
    let cancelled = false;
    setResolved("");
    setDirs([]);
    setFiles([]);
    setParent(null);
    setError("");
    const q = path ? `?path=${encodeURIComponent(path)}` : "";
    fetch(`/api/browse${q}`, { signal: controller.signal })
      .then((r) =>
        r.ok
          ? r.json()
          : Promise.reject(new Error(r.status === 404 ? "No folder at that path." : `Can't open this folder (${r.status}).`)),
      )
      .then((d) => {
        if (cancelled) return;
        setResolved(d.path);
        setTyped(d.path);
        setDirs(d.dirs);
        setFiles(d.files ?? []);
        setParent(d.parent);
      })
      .catch((e) => {
        if (!cancelled && (e as Error).name !== "AbortError") setError((e as Error).message);
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [path, reload]);

  return (
    <div className="modal-back" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()}>
        <div className="modal-head">{title}</div>
        <input
          className="modal-path"
          value={typed}
          spellCheck={false}
          placeholder="/path/to/a/folder"
          title="Type or paste a path, then press Enter"
          onChange={(e) => setTyped(e.target.value)}
          onKeyDown={(e) => {
            if (e.key !== "Enter" || !typed.trim()) return;
            setPath(typed.trim());
            setReload((n) => n + 1);
          }}
        />
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
