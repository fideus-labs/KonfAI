// SPDX-License-Identifier: Apache-2.0

import { useEffect, useState } from "react";
import { useJson } from "./useJson";

// KonfAI Scout's reports for this experiment: what a vision model found wrong, where (in millimetres), with the
// views and measurements that back it. "Go to" loads the case's image in the viewer and puts the crosshair on
// the observation.

type Box = { min: number[]; max: number[] };
type Observation = {
  id: string;
  kind: string;
  defect: string;
  severity: string;
  confidence: number;
  description: string;
  box_mm: Box | null;
  plane: string | null;
  views: string[];
  measurements: string[];
  notes: string[];
};
type Measurement = { id: string; kind: string; reference: boolean; values: Record<string, number | string | null> };
type ReportSummary = {
  name: string;
  task: string | null;
  case: string | null;
  verdict: string | null;
  confidence: number | null;
  observations: number;
  created_at: string | null;
  mode: string | null;
  model: string | null;
  error: string | null;
};
type Report = {
  name: string;
  task: string;
  case: string;
  image: string | null;
  truth_available: boolean;
  conclusion: { verdict: string; summary: string; confidence: number; by: string; reason: string | null } | null;
  observations: Observation[];
  measurements: Measurement[];
  views: number;
  usage: { images_sent?: number; input_tokens?: number; output_tokens?: number; cost_usd?: number | null } | null;
  budget: { exhausted?: string | null } | null;
  error: string | null;
  provider: { name: string; model: string } | null;
  mode: string | null;
};

export function boxCentre(box: Box): [number, number, number] {
  return [0, 1, 2].map((i) => (box.min[i] + box.max[i]) / 2) as [number, number, number];
}

function mm(values: number[]): string {
  return values.map((v) => v.toFixed(1)).join(", ");
}

function kindLabel(kind: string): string {
  return { measured_property: "measured", suspected_error: "suspected", confirmed_error: "confirmed" }[kind] ?? kind;
}

export default function Scout({
  session,
  onGoTo,
}: {
  session: string;
  onGoTo: (image: string, mm: [number, number, number]) => void;
}) {
  const [retry, setRetry] = useState(0);
  const { data, loading, error } = useJson<{ reports: ReportSummary[] }>(
    `/api/scout/reports?session=${encodeURIComponent(session)}`,
    [session, retry],
  );
  const reports = data?.reports ?? [];
  const [selected, setSelected] = useState<string | null>(null);
  const { data: report, error: failed } = useJson<Report>(
    selected ? `/api/scout/report?session=${encodeURIComponent(session)}&name=${encodeURIComponent(selected)}` : "",
    [selected, session],
  );
  const [shown, setShown] = useState<string | null>(null); // an evidence image enlarged

  useEffect(() => {
    if (!selected && reports.length) setSelected(reports[0].name);
  }, [reports, selected]);

  const evidence = (view: string) =>
    `/api/scout/evidence?session=${encodeURIComponent(session)}&name=${encodeURIComponent(selected ?? "")}&view=${view}`;

  return (
    <div className="scout">
      <div className="feed-section">
        <span className="feed-label">Scout · visual inspections</span>
        <button className="rtab" onClick={() => setRetry((n) => n + 1)} title="Reload the list">
          ↻
        </button>
      </div>
      {loading && !reports.length && <div className="feed-sub">Loading…</div>}
      {error && <div className="feed-sub">Could not list the reports: {error}</div>}
      {!loading && !error && !reports.length && (
        <div className="feed-sub">
          No inspection yet. Ask the assistant to inspect a result with Scout (run_scout_inspection), or run
          <code> konfai-scout inspect</code> with this session's Scout folder as output.
        </div>
      )}
      {reports.length > 0 && (
        <div className="scout-list">
          {reports.map((r) => (
            <button
              key={r.name}
              className={r.name === selected ? "scout-row on" : "scout-row"}
              onClick={() => setSelected(r.name)}
              title={`${r.task ?? ""} · ${r.mode ?? ""} · ${r.model ?? ""}`}
            >
              <span className={`scout-verdict ${r.verdict ?? "none"}`}>{r.verdict ?? (r.error ? "error" : "…")}</span>
              <span className="scout-name">{r.name}</span>
              <span className="feed-sub">{r.observations} observation{r.observations === 1 ? "" : "s"}</span>
            </button>
          ))}
        </div>
      )}
      {failed && <div className="feed-sub">Could not read the report: {failed}</div>}
      {report && (
        <div className="scout-report">
          {report.conclusion && (
            <div className="scout-conclusion">
              <b>
                {report.conclusion.verdict} ({Math.round(report.conclusion.confidence * 100)}%)
              </b>{" "}
              {report.conclusion.summary}
              {report.conclusion.reason && <div className="feed-sub">{report.conclusion.reason}</div>}
            </div>
          )}
          {report.error && <div className="feed-sub">Error: {report.error}</div>}
          {!report.truth_available && (
            <div className="feed-sub">No reference was given: nothing here is measured against a truth.</div>
          )}
          {report.budget?.exhausted && <div className="feed-sub">Budget exhausted: {report.budget.exhausted}.</div>}
          <div className="feed-sub">
            {report.provider?.name}/{report.provider?.model} · {report.mode} · {report.views} views ·{" "}
            {report.measurements.length} measurements
            {report.usage?.cost_usd != null && <> · ${report.usage.cost_usd.toFixed(4)}</>}
          </div>
          {report.observations.length === 0 && <div className="feed-sub">No observation recorded.</div>}
          {report.observations.map((o) => (
            <div key={o.id} className={`scout-obs sev-${o.severity}`}>
              <div className="scout-obs-head">
                <span className={`scout-kind ${o.kind}`}>{kindLabel(o.kind)}</span>
                <b>{o.defect}</b>
                <span className="feed-sub">
                  {o.severity} · {Math.round(o.confidence * 100)}%
                </span>
                {o.box_mm && report.image && (
                  <button
                    className="rtab"
                    onClick={() => onGoTo(report.image!, boxCentre(o.box_mm!))}
                    title={`Load ${report.image} and put the crosshair at ${mm(boxCentre(o.box_mm))} mm`}
                  >
                    Go to
                  </button>
                )}
              </div>
              <div>{o.description}</div>
              {o.box_mm && <div className="feed-sub">centre {mm(boxCentre(o.box_mm))} mm · size {mm([0, 1, 2].map((i) => o.box_mm!.max[i] - o.box_mm!.min[i]))} mm</div>}
              {o.notes.map((n) => (
                <div key={n} className="feed-sub">
                  {n}
                </div>
              ))}
              {o.measurements.map((id) => {
                const m = report.measurements.find((x) => x.id === id);
                return (
                  m && (
                    <div key={id} className="feed-sub">
                      {id} ({m.reference ? "against the reference" : "no reference"}):{" "}
                      {Object.entries(m.values)
                        .map(([k, v]) => `${k} = ${v == null ? "n/a" : typeof v === "number" ? v.toPrecision(4) : v}`)
                        .join(", ")}
                    </div>
                  )
                );
              })}
              <div className="scout-evidence">
                {o.views.map((v) => (
                  <img key={v} src={evidence(v)} alt={v} title={v} onClick={() => setShown(shown === v ? null : v)} />
                ))}
              </div>
            </div>
          ))}
          {shown && (
            <div className="scout-zoom" onClick={() => setShown(null)}>
              <img src={evidence(shown)} alt={shown} />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
