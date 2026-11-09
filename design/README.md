# KonfAI identity kit

One visual identity for the family: KonfAI (the engine), KonfAI Apps, KonfAI Edge, KonfAI Studio, KonfAI MCP and
KonfAI Scout, and FidOpt. It sits between a console and a landing page: a dark
ground that is never a neutral grey, Archivo for the titles, IBM Plex for the text and the data, one hue per
piece, one clock for every motion. `design/` is the source; `sync.sh` copies it to the places that serve it.

| File | What it holds |
|---|---|
| `fonts.css`, `fonts/` | The three faces, served from the site itself (no third-party font host): Archivo 100–900 (variable), IBM Plex Sans 400/500/600, IBM Plex Mono 400/500, latin subsets, with their OFL licences |
| `tokens.css` | Every value, as `--k-*` custom properties on `:root`: hues, grounds, ink, type, scale, space, radii, controls, motion |
| `components.css` | The primitives on those tokens: the `.hue-*` classes, `.k-kicker`, `.k-btn`, `.k-card`, `.k-chips`, `.k-rule`, `.k-reveal` |
| `volume-hero.js` | The hero: three slice planes drifting in a wire cube and a patch sweeping them, on a canvas; an ES module with one exported function |
| `sync.sh` | Copies the kit to `site/design/` (everything) and `konfai-studio/frontend/src/design/` (fonts and tokens) |

## The tokens

Hues, one per piece. The solid is for fills, rails, marks and meters; the `-ink` is the same hue as text on a
dark ground (every ink reads at 10:1 or better on `--k-night`); the `-deep` is the hue on paper.

| Piece | Solid | Ink | Deep (light theme) | Also means |
|---|---|---|---|---|
| Engine | `--k-sage` | `--k-sage-ink` | `--k-sage-deep` | what is current, what is ready |
| Apps | `--k-teal` | `--k-teal-ink` | `--k-teal-deep` | |
| Edge | `--k-violet` | `--k-violet-ink` | `--k-violet-deep` | |
| Studio | `--k-amber` | `--k-amber-ink` | `--k-amber-deep` | the live signal: a tool running, a job writing |
| MCP | `--k-steel` | `--k-steel-ink` | `--k-steel-deep` | a measured fact |
| Scout | `--k-coral` | `--k-coral-ink` | `--k-coral-deep` | an error, a confirmed defect |

Grounds: `--k-night`, `--k-night-2`, `--k-night-3` (a page), `--k-graphite`, `--k-graphite-2`, `--k-graphite-3`
(a tool's chrome), `--k-paper`, `--k-paper-2`, `--k-paper-3` (a light theme). Ink on the night: `--k-ink`,
`--k-ink-2`, `--k-muted`, `--k-dim`; hairlines `--k-line`, `--k-line-2`.

Type: `--k-display` (Archivo, titles only, at `--k-display-weight` 680 and `--k-display-tracking`), `--k-sans`
(IBM Plex Sans, text), `--k-mono` (IBM Plex Mono, data, labels, code). Scale `--k-fs-0` … `--k-fs-7`: six fixed
steps for an interface (10.5, 12, 13.5, 15, 20, 28 px), two fluid steps for a page (a section title, a headline).

Space `--k-s-1` … `--k-s-7` (4 to 48 px); radii `--k-r-1` 6, `--k-r-2` 10, `--k-r-3` 14, `--k-r-pill`;
controls `--k-ctl` 32 px, `--k-ctl-touch` 40 px; a page's measure `--k-max`. Motion: `--k-dur` 150 ms for a
state, `--k-dur-2` 400 ms for a reveal, `--k-ease` a hard ease-out; both durations are 0 under
`prefers-reduced-motion`.

## On a page

```html
<link rel="stylesheet" href="design/fonts.css">
<link rel="stylesheet" href="design/tokens.css">
<link rel="stylesheet" href="design/components.css">
<link rel="stylesheet" href="your-page.css">
```

Give the page, a section or a card its hue with a class: `hue-engine`, `hue-apps`, `hue-edge`, `hue-studio`,
`hue-mcp`, `hue-scout`. Everything below reads `--k-hue` and `--k-hue-ink`: the kicker, the card's rail, the
ghost button's hover, the chips' hover, the rule. Your own stylesheet maps its semantic names onto the tokens
(`--bg: var(--k-night)`, `--link: var(--k-hue-ink)`) and never writes a colour of its own.

The hero, where a page has one:

```html
<section class="hero"><canvas id="volume" aria-hidden="true"></canvas> …</section>
<script type="module">
  import { mountVolumeHero } from "./design/volume-hero.js";
  mountVolumeHero(document.getElementById("volume"));
</script>
```

The canvas fills its positioned parent; the options (plane hues, sweep hue, ground, fps, grid density, scale,
centre) are in the module's header. A reveal is a block with `k-reveal`; a script puts `js` on `<html>` and `in`
on the block when it scrolls into view (`site/landing.js` does, with a 60 ms stagger per column and a count-up on
`[data-count]` figures). Without the script every block shows.

## In Studio

`konfai-studio/frontend/src/styles.css` imports `./design/fonts.css` and `./design/tokens.css` (Vite inlines them
at build and copies the fonts beside the bundle, so Studio serves its own fonts) and maps its semantic names:
`--fsans`, `--fmono`, `--fdisplay` onto the three faces; `--sage`, `--working`, `--err`, `--kind-*`, `--verdict-*`
onto the hues (the `-deep` hues in the light theme); `--r-*`, `--dur`, `--ease` onto the radii and the clock.
Studio keeps its own graphite surfaces and ink steps, measured against its panels; Archivo carries the verdict
word and the welcome title only.

## Rules

- The six hues and nothing else. A new meaning takes an existing hue (the table says what each one also means),
  never a seventh colour.
- Archivo is for titles: a headline, a section title, a card title, a verdict. Text is Plex Sans; anything a
  machine wrote (a path, a figure, a label, code) is Plex Mono.
- A page sits on the night; a tool sits on graphite or paper. Nothing is a neutral grey.
- Fonts are served from the site; no page loads a font from a third-party host.
- One clock: 150 ms for a state, 400 ms for a reveal, the same ease; nothing moves under reduced motion.
- Edit `design/`, run `design/sync.sh`, commit the copies with the change.

Fonts under the SIL Open Font License 1.1: `fonts/OFL-archivo.txt`, `fonts/OFL-ibmplexsans.txt`,
`fonts/OFL-ibmplexmono.txt`.
