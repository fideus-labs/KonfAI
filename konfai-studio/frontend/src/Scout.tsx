// SPDX-License-Identifier: Apache-2.0

import { useEffect, useMemo, useState } from "react";
import { useJson } from "./useJson";

// KonfAI Scout's reports for this experiment, read like a radiologist's sheet: the verdict first, then every
// observation with its kind, where it sits (millimetres), what was measured there and the views that show it.
// "Go to" loads the case's image in the viewer and puts the crosshair on the observation.

type Box = { min: number[]; max: number[] };
type SecondReading = { answer?: string | null; sentence?: string | null; views?: string[] } | null;
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
  second_reading?: SecondReading;
};
type Measurement = {
  id: string;
  kind: string;
  reference: boolean;
  values: Record<string, number | string | null>;
  box_measured?: Box | null;
};
type ViewMeta = {
  id: string;
  kind: string | null;
  plane: string | null;
  index: number | null;
  position_mm: number[] | null;
  mode: string | null;
  sent: boolean | null;
  size_px: number[] | null;
};
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
  views_meta?: ViewMeta[];
  usage: {
    images_sent?: number;
    input_tokens?: number;
    output_tokens?: number;
    cost_usd?: number | null;
    cost_known?: boolean;
  } | null;
  budget: {
    exhausted?: string | null;
    limits?: Record<string, number | null>;
    spent?: Record<string, number | null>;
  } | null;
  error: string | null;
  provider: { name: string; model: string } | null;
  mode: string | null;
  created_at?: string | null;
  duration_s?: number | null;
  second_reading?: { model?: string; answers?: Record<string, number> } | null;
  report_md: string;
};

export function boxCentre(box: Box): [number, number, number] {
  return [0, 1, 2].map((i) => (box.min[i] + box.max[i]) / 2) as [number, number, number];
}

function mm(values: number[]): string {
  return values.map((v) => v.toFixed(1)).join(", ");
}

const KIND_LABEL: Record<string, string> = {
  measured_property: "measured",
  suspected_error: "suspected error",
  confirmed_error: "confirmed error",
};

const VERDICT_LABEL: Record<string, string> = {
  defective: "Defective",
  acceptable: "Acceptable",
  insufficient_evidence: "Insufficient evidence",
};

function words(value: string): string {
  return value.replace(/_/g, " ");
}

function when(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

function duration(s: number | null | undefined): string {
  if (s == null) return "";
  return s < 90 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

// The coordinate a slice is cut at, along the axis its plane is normal to: what a reader asks first.
function cutAt(view: ViewMeta): string {
  if (!view.position_mm) return "";
  const axis = { sagittal: 0, coronal: 1, axial: 2 }[view.plane ?? ""];
  if (axis == null) return `(${mm(view.position_mm)}) mm`;
  return `${"xyz"[axis]} = ${view.position_mm[axis].toFixed(1)} mm`;
}

function caption(view: ViewMeta | undefined, id: string): string {
  if (!view) return id;
  const parts = [id, view.plane, view.mode === "both" ? "image + prediction" : words(view.mode ?? "")];
  if (view.kind === "montage") parts.push("montage");
  const cut = cutAt(view);
  if (cut) parts.push(cut);
  return parts.filter(Boolean).join(" · ");
}

// KonfAI's metric keys read "OUT:TRUTH:Dice:1"; the reader wants "Dice, label 1". Dotted statistics stay as is.
function metricLabel(key: string): string {
  if (!key.includes(":")) return key.replace(/\./g, " · ");
  const parts = key.split(":");
  const label = parts.length >= 4 ? parts.slice(3).join(":") : null;
  return label != null ? `${parts[2]}, label ${label}` : parts[parts.length - 1];
}

function formatValue(value: number | string | null): string {
  if (value == null) return "n/a";
  if (typeof value !== "number") return String(value);
  if (Number.isInteger(value)) return value.toLocaleString();
  return Math.abs(value) >= 100 ? value.toFixed(1) : value.toPrecision(4);
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
  const [wantMd, setWantMd] = useState(false);
  const { data: md } = useJson<{ content?: string; text?: string }>(
    wantMd && selected
      ? `/api/experiment/file?session=${encodeURIComponent(session)}&path=${encodeURIComponent(`Scout/${selected}/report.md`)}`
      : "",
    [wantMd, selected, session],
  );

  useEffect(() => {
    if (!selected && reports.length) setSelected(reports[0].name);
  }, [reports, selected]);
  useEffect(() => setWantMd(false), [selected]);

  const evidence = (view: string) =>
    `/api/scout/evidence?session=${encodeURIComponent(session)}&name=${encodeURIComponent(selected ?? "")}&view=${view}`;

  const viewsById = useMemo(() => new Map((report?.views_meta ?? []).map((v) => [v.id, v] as const)), [report]);
  const measurementsById = useMemo(
    () => new Map((report?.measurements ?? []).map((m) => [m.id, m] as const)),
    [report],
  );
  const uncited = useMemo(() => {
    if (!report) return [];
    const cited = new Set(report.observations.flatMap((o) => o.views));
    return (report.views_meta ?? []).filter((v) => v.sent !== false && !cited.has(v.id));
  }, [report]);

  const verdict = report?.conclusion?.verdict ?? (report?.error ? "error" : "none");
  const spent = report?.budget?.spent ?? {};
  const limits = report?.budget?.limits ?? {};

  return (
    <div className="scout">
      <div className="scout-top">
        <span className="scout-eyebrow">Scout · visual inspections</span>
        {reports.length > 1 && (
          <div className="scout-tabs" role="tablist" aria-label="Inspections of this experiment">
            {reports.map((r) => (
              <button
                key={r.name}
                role="tab"
                aria-selected={r.name === selected}
                className={r.name === selected ? "scout-tab on" : "scout-tab"}
                onClick={() => setSelected(r.name)}
                title={`${r.task ?? ""} · ${r.mode ?? ""} · ${r.model ?? ""} · ${when(r.created_at)}`}
              >
                <i className={`verdict-dot ${r.verdict ?? (r.error ? "error" : "none")}`} aria-hidden="true" />
                {r.name}
              </button>
            ))}
          </div>
        )}
        <button className="scout-reload" onClick={() => setRetry((n) => n + 1)} title="Reload the list">
          ↻
        </button>
      </div>

      {loading && !reports.length && <div className="scout-empty">Loading…</div>}
      {error && <div className="scout-empty">Could not list the reports: {error}</div>}
      {!loading && !error && !reports.length && (
        <div className="scout-empty">
          <b>No inspection yet.</b> Ask the assistant to inspect a result with Scout (<code>run_scout_inspection</code>
          ), or run <code>konfai-scout inspect</code> with this session's Scout folder as output.
        </div>
      )}
      {failed && <div className="scout-empty">Could not read the report: {failed}</div>}

      {report && (
        <article className="sheet" aria-label={`Scout report ${report.name}`}>
          <header className={`verdict-band ${verdict}`}>
            <div className="verdict-word">{VERDICT_LABEL[verdict] ?? words(verdict)}</div>
            {report.conclusion && (
              <div
                className="verdict-meter"
                role="meter"
                aria-valuemin={0}
                aria-valuemax={100}
                aria-valuenow={Math.round(report.conclusion.confidence * 100)}
                aria-label="Confidence"
                title={`Confidence ${Math.round(report.conclusion.confidence * 100)} %`}
              >
                <i style={{ width: `${Math.round(report.conclusion.confidence * 100)}%` }} />
                <span>{Math.round(report.conclusion.confidence * 100)} %</span>
              </div>
            )}
            <dl className="verdict-facts">
              <dt>Case</dt>
              <dd>{report.case}</dd>
              <dt>Task</dt>
              <dd>{report.task}</dd>
              {report.provider && (
                <>
                  <dt>Model</dt>
                  <dd>
                    {report.provider.model} <span className="dim">via {report.provider.name}</span>
                  </dd>
                </>
              )}
              {report.mode && (
                <>
                  <dt>Mode</dt>
                  <dd>{report.mode}</dd>
                </>
              )}
              {report.created_at && (
                <>
                  <dt>When</dt>
                  <dd>
                    {when(report.created_at)}
                    {report.duration_s != null && <span className="dim"> · {duration(report.duration_s)}</span>}
                  </dd>
                </>
              )}
              <dt>Budget</dt>
              <dd>
                {spent.tool_calls ?? "?"}
                {limits.max_tool_calls != null && <span className="dim">/{limits.max_tool_calls}</span>} calls ·{" "}
                {spent.images ?? report.usage?.images_sent ?? "?"}
                {limits.max_images != null && <span className="dim">/{limits.max_images}</span>} images
                {report.usage?.cost_usd != null && <> · ${report.usage.cost_usd.toFixed(2)}</>}
                {report.budget?.exhausted && <span className="warn"> · exhausted: {report.budget.exhausted}</span>}
              </dd>
              {report.conclusion && report.conclusion.by !== "model" && (
                <>
                  <dt>Concluded by</dt>
                  <dd>{report.conclusion.by}</dd>
                </>
              )}
            </dl>
          </header>

          {report.conclusion && (
            <section className="lead">
              {report.conclusion.summary.split(/\n{2,}/).map((para, i) => (
                <p key={i}>{para}</p>
              ))}
              {report.conclusion.reason && <p className="dim">{report.conclusion.reason}</p>}
            </section>
          )}
          {report.error && <div className="sheet-note err">The inspection ended on an error: {report.error}</div>}
          {!report.truth_available && (
            <div className="sheet-note">No reference was given: nothing here is measured against a truth.</div>
          )}

          {report.observations.length === 0 && <div className="sheet-note">No observation recorded.</div>}
          {report.observations.length > 0 && (
            <ol className="obs-list">
              {report.observations.map((o) => (
                <li key={o.id} className={`obs ${o.kind} sev-${o.severity}`}>
                  <header className="obs-head">
                    <span className="obs-id">{o.id}</span>
                    <span className="kind-badge">{KIND_LABEL[o.kind] ?? words(o.kind)}</span>
                    <b className="obs-defect">{words(o.defect)}</b>
                    <span className="obs-sev" title="severity · confidence">
                      {o.severity} · {Math.round(o.confidence * 100)} %
                    </span>
                    {o.box_mm && report.image && (
                      <button
                        className="obs-goto"
                        onClick={() => onGoTo(report.image!, boxCentre(o.box_mm!))}
                        title={`Load ${report.image} and put the crosshair at ${mm(boxCentre(o.box_mm))} mm`}
                      >
                        Go to
                      </button>
                    )}
                  </header>
                  <p className="obs-text">{o.description}</p>
                  {o.box_mm && (
                    <div className="obs-where">
                      {o.plane && <span>{o.plane}</span>}
                      <span>centre {mm(boxCentre(o.box_mm))} mm</span>
                      <span>size {mm([0, 1, 2].map((i) => o.box_mm!.max[i] - o.box_mm!.min[i]))} mm</span>
                    </div>
                  )}
                  {o.notes.map((n) => (
                    <div key={n} className="obs-note">
                      {n}
                    </div>
                  ))}
                  {o.second_reading && o.second_reading.answer && (
                    <div className={`second-reading ${o.second_reading.answer}`}>
                      <b>Blind second reading: {words(o.second_reading.answer)}.</b> {o.second_reading.sentence}
                    </div>
                  )}
                  {o.measurements.length > 0 && (
                    <table className="obs-measures">
                      <thead>
                        <tr>
                          <th>Measured</th>
                          <th>Value</th>
                          <th>Against</th>
                        </tr>
                      </thead>
                      <tbody>
                        {o.measurements.flatMap((id) => {
                          const m = measurementsById.get(id);
                          if (!m) return [];
                          const rows = Object.entries(m.values);
                          return rows.slice(0, 6).map(([key, value], i) => (
                            <tr key={`${id}-${key}`}>
                              <td>
                                {i === 0 && <span className="m-id">{id}</span>} <span title={key}>{metricLabel(key)}</span>
                              </td>
                              <td className="num">{formatValue(value)}</td>
                              <td>{i === 0 ? (m.reference ? "the truth" : "nothing (descriptive)") : ""}</td>
                            </tr>
                          ));
                        })}
                      </tbody>
                    </table>
                  )}
                  {o.views.length > 0 && (
                    <div className="filmstrip">
                      {o.views.map((v) => (
                        <figure key={v}>
                          <img
                            src={evidence(v)}
                            alt={caption(viewsById.get(v), v)}
                            loading="lazy"
                            decoding="async"
                            onClick={() => setShown(v)}
                          />
                          <figcaption>{caption(viewsById.get(v), v)}</figcaption>
                        </figure>
                      ))}
                    </div>
                  )}
                </li>
              ))}
            </ol>
          )}

          {uncited.length > 0 && (
            <details className="uncited">
              <summary>
                {uncited.length} more view{uncited.length === 1 ? "" : "s"} the model looked at, cited by no observation
              </summary>
              <div className="filmstrip">
                {uncited.map((v) => (
                  <figure key={v.id}>
                    <img
                      src={evidence(v.id)}
                      alt={caption(v, v.id)}
                      loading="lazy"
                      decoding="async"
                      onClick={() => setShown(v.id)}
                    />
                    <figcaption>{caption(v, v.id)}</figcaption>
                  </figure>
                ))}
              </div>
            </details>
          )}

          <footer className="sheet-foot">
            <details onToggle={(e) => (e.currentTarget as HTMLDetailsElement).open && setWantMd(true)}>
              <summary>The report as written, report.md</summary>
              {md ? <pre className="sheet-md">{md.content ?? md.text ?? ""}</pre> : <div className="dim">Loading…</div>}
              <code className="dim">{report.report_md}</code>
            </details>
            {report.usage && (
              <span className="dim">
                {report.usage.input_tokens?.toLocaleString()} in · {report.usage.output_tokens?.toLocaleString()} out
                tokens
                {report.second_reading?.model && <> · second reading by {report.second_reading.model}</>}
              </span>
            )}
          </footer>
        </article>
      )}

      {shown && selected && (
        <div className="scout-zoom" onClick={() => setShown(null)} role="dialog" aria-label={`View ${shown}`}>
          <img src={evidence(shown)} alt={caption(viewsById.get(shown), shown)} />
          <div className="scout-zoom-cap">{caption(viewsById.get(shown), shown)}</div>
        </div>
      )}
    </div>
  );
}
