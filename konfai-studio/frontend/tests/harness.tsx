// SPDX-License-Identifier: Apache-2.0
//
// Mounts one component of the app alone, with the props a test hands it: the real code, the real
// vite build, the API answered by the test's routes. `window.__mount(name, props)` re-renders in
// place, so a second call is a prop change and not a fresh mount.
import React, { Suspense, lazy, useState } from "react";
import { createRoot } from "react-dom/client";
import FolderBrowser from "../src/FolderBrowser";
import RightPanel from "../src/RightPanel";
import { useJobStream, type JobStream } from "../src/useJobStream";
import { useJson } from "../src/useJson";
import "../src/styles.css";

const Viewer = lazy(() => import("../src/Viewer"));

function FolderPicker() {
  const [open, setOpen] = useState(false);
  const [picked, setPicked] = useState("");
  return <>
    <button onClick={() => setOpen(true)}>Browse folders</button>
    <button>Outside control</button>
    <output>{picked}</output>
    {open && <FolderBrowser start="/data" onClose={() => setOpen(false)} onPick={(path) => {
      setPicked(path);
      setOpen(false);
    }} />}
  </>;
}

function PanelProbe({ stream, session = "S" }: { stream: JobStream; session?: string }) {
  const [path, setPath] = useState<string | null>(null);
  return <section className="right-panel" style={{ height: "100vh" }}>
    <RightPanel session={session} stream={stream} volumePath={path} onVolumePathChange={setPath}
      comparePath={null} onComparePathChange={() => undefined} />
  </section>;
}

function JsonProbe({ url }: { url: string }) {
  const { data, loading, error } = useJson<unknown>(url, [url]);
  return <pre id="json">{JSON.stringify({ url, data, loading, error })}</pre>;
}

function StreamProbe({ session }: { session: string }) {
  const stream = useJobStream(session, 0);
  return <pre id="stream">{JSON.stringify({ status: stream.status, run: stream.run, lines: stream.lines.length })}</pre>;
}

const components: Record<string, (props: any) => React.ReactElement> = {
  PanelProbe: (props) => <PanelProbe {...props} />,
  FolderPicker: () => <FolderPicker />,
  Viewer: (props) => <Suspense><Viewer path={props.path ?? null} onPathChange={() => undefined} /></Suspense>,
  FolderBrowser: (props) => (
    <FolderBrowser
      start={props.start ?? ""}
      onPickFile={props.pickFile ? (p) => ((window as any).__picked = p) : undefined}
      onClose={() => ((window as any).__closed = ((window as any).__closed ?? 0) + 1)}
    />
  ),
  JsonProbe: (props) => <JsonProbe url={props.url} />,
  StreamProbe: (props) => <StreamProbe session={props.session} />,
};

const root = createRoot(document.getElementById("root")!);
(window as any).__mount = (name: string, props: Record<string, unknown> = {}) => {
  root.render(React.createElement(components[name], props));
};
(window as any).__unmount = () => root.render(null);
