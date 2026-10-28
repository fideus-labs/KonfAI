// SPDX-License-Identifier: Apache-2.0
//
// The part of the KonfAI Edge engine's API (its wasm-bindgen `konfai_edge.js`) that Studio's Deploy pane
// calls. The engine itself ships in each app's `web/` folder; these declarations are copied from the
// `konfai_edge.d.ts` of the build the README names.

/** An app as its `app.json` and the files beside it describe it, decided as konfai-apps decides it. */
export class App {
  free(): void;
  /** `app_json` is the app's `app.json`, `files` the programs its `web/` directory holds (`prediction.json`...). */
  constructor(name: string, app_json: string, files: string[]);
  /** JSON: `inputs`, `outputs`, `bounds` (`models`, `tta`, `mc_dropout`), `prediction`, `uncertainty`, `evaluations`. */
  describe(): string;
  /** `options` (JSON `ensemble`, `ensemble_models`, `tta`, `mc`, `uncertainty`) as JSON `{program, checkpoints, tta, notes}`. */
  run(options: string): string;
}

export class Predictor {
  free(): void;
  /** The members `checkpoints` and `tta` pick of `program`; `names` and `weights` hold what `selected_weights` lists. */
  static selected(
    program: string,
    checkpoints: string[],
    tta: number,
    names: string[],
    weights: Uint8Array[],
    device: string,
    precision?: string | null,
    budget?: number | null,
    device_budget?: number | null,
  ): Predictor;
  /** Run the program on the dataset files `paths` name and `contents` hold: resolves to every entry it writes. */
  predict(paths: string[], contents: Uint8Array[]): Promise<any>;
}

/** Burn's WebGPU device, initialised before a model is built on it. */
export function init_webgpu(): Promise<any>;
/** The message of the panic that aborted a call, empty while none has. */
export function last_panic(): string;
/** Hand the WebGPU device's memory pool back. */
export function release_device_memory(): void;
/** The `<stem>.bpk` weight files a run of `program` needs on the members `checkpoints` and `tta` pick. */
export function selected_weights(program: string, checkpoints: string[], tta: number): string[];
/** The volume `bytes` hold, a `source_format` file, as the bytes of a `target_format` file. */
export function transcode_volume(bytes: Uint8Array, source_format: string, target_format: string, max_bytes: number): Uint8Array;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;
export default function __wbg_init(module_or_path?: { module_or_path: InitInput | Promise<InitInput> }): Promise<unknown>;
