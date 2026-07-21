import { useEffect, useRef, useState } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";

// The bottom drawer is a real login shell rooted at the workspace — run nvidia-smi, activate an env,
// inspect files. The job log lives in the Live tab now, so this is a general-purpose terminal.
export default function Console() {
  const [open, setOpen] = useState(false);
  const [connected, setConnected] = useState(false);
  const hostRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  // Spin up the shell the first time the drawer opens (lazy), then keep it alive across collapses —
  // the host is only hidden, so the session and scrollback survive.
  useEffect(() => {
    if (!open || termRef.current || !hostRef.current) return;
    const term = new Terminal({
      fontFamily: "ui-monospace, 'JetBrains Mono', Menlo, Consolas, monospace",
      fontSize: 12.5,
      cursorBlink: true,
      theme: { background: "#161615", foreground: "#e7e6e1", cursor: "#d6d5cf", selectionBackground: "#3a3a37" },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(hostRef.current);
    fit.fit();
    termRef.current = term;
    fitRef.current = fit;

    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/api/terminal`);
    ws.binaryType = "arraybuffer";
    wsRef.current = ws;
    ws.onopen = () => {
      setConnected(true);
      ws.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
    };
    ws.onmessage = (e) => term.write(typeof e.data === "string" ? e.data : new Uint8Array(e.data));
    ws.onclose = () => {
      setConnected(false);
      term.write("\r\n\x1b[2m[shell ended — reopen the drawer to start a new one]\x1b[0m\r\n");
      termRef.current = null; // let a reopen spawn a fresh shell
    };
    term.onData((d) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: "input", data: d })));
    term.onResize(({ cols, rows }) => ws.readyState === WebSocket.OPEN && ws.send(JSON.stringify({ type: "resize", cols, rows })));
  }, [open]);

  // Teardown only when the whole app unmounts.
  useEffect(
    () => () => {
      wsRef.current?.close();
      termRef.current?.dispose();
    },
    [],
  );

  // Fit to the drawer whenever it opens or the window resizes.
  useEffect(() => {
    if (!open) return;
    const refit = () => {
      fitRef.current?.fit();
      termRef.current?.focus();
    };
    const id = window.setTimeout(refit, 60);
    window.addEventListener("resize", refit);
    return () => {
      window.clearTimeout(id);
      window.removeEventListener("resize", refit);
    };
  }, [open]);

  return (
    <section className={open ? "console open" : "console"}>
      <div className="console-head" onClick={() => setOpen((o) => !o)}>
        <button className="ctab on">Terminal</button>
        {open && <span className={connected ? "cst running" : "cst"}>{connected ? "connected" : "…"}</span>}
        <span className="cbar-spacer" />
        <button className="toggle">{open ? "▾" : "▸"}</button>
      </div>
      <div className="term-host" ref={hostRef} style={{ display: open ? "block" : "none" }} />
    </section>
  );
}
