// SPDX-License-Identifier: Apache-2.0
//
// The few drawn icons the shell needs where a glyph read as a costume: one stroke width, round caps,
// sized by the control that holds them (currentColor).
type P = { size?: number };

const base = (size: number) => ({
  viewBox: "0 0 24 24",
  width: size,
  height: size,
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.8,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
});

export function MenuIcon({ size = 18 }: P) {
  return (
    <svg {...base(size)}>
      <line x1="4" y1="7" x2="20" y2="7" />
      <line x1="4" y1="12" x2="20" y2="12" />
      <line x1="4" y1="17" x2="20" y2="17" />
    </svg>
  );
}

export function ChevronIcon({ open, size = 14 }: P & { open: boolean }) {
  return (
    <svg {...base(size)} style={{ transform: open ? "rotate(90deg)" : undefined, transition: "transform var(--dur) var(--ease)" }}>
      <polyline points="9 6 15 12 9 18" />
    </svg>
  );
}

export function CloseIcon({ size = 14 }: P) {
  return (
    <svg {...base(size)}>
      <line x1="6" y1="6" x2="18" y2="18" />
      <line x1="18" y1="6" x2="6" y2="18" />
    </svg>
  );
}

export function RefreshIcon({ size = 14 }: P) {
  return (
    <svg {...base(size)}>
      <path d="M20 12a8 8 0 1 1-2.6-5.9" />
      <polyline points="20 4 20 9 15 9" />
    </svg>
  );
}

export function ExpandIcon({ size = 13 }: P) {
  return (
    <svg {...base(size)}>
      <polyline points="15 4 20 4 20 9" />
      <polyline points="9 20 4 20 4 15" />
      <line x1="20" y1="4" x2="14" y2="10" />
      <line x1="4" y1="20" x2="10" y2="14" />
    </svg>
  );
}

export function SlidersIcon({ size = 16 }: P) {
  return (
    <svg {...base(size)}>
      <line x1="4" y1="7" x2="20" y2="7" />
      <line x1="4" y1="17" x2="20" y2="17" />
      <circle cx="9" cy="7" r="2.2" fill="var(--panel-2)" />
      <circle cx="15" cy="17" r="2.2" fill="var(--panel-2)" />
    </svg>
  );
}

export function TerminalIcon({ size = 16 }: P) {
  return (
    <svg {...base(size)}>
      <rect x="3" y="5" width="18" height="14" rx="2" />
      <polyline points="7 9 10 12 7 15" />
      <line x1="12" y1="15" x2="17" y2="15" />
    </svg>
  );
}
