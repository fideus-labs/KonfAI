// The landing's three dynamic pieces: the hero canvas, the search over the site's index, the phone menu.
// No dependency. Generated index below; regenerate it with the site's build note when pages change.
(function () {
  "use strict";

  // ---- hero canvas: three orthogonal slice planes of a volume, drifting; a patch sweeping over them ----
  var canvas = document.getElementById("volume");
  if (canvas && canvas.getContext) {
    var ctx = canvas.getContext("2d");
    var reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;
    var W = 0, H = 0, dpr = 1, raf = 0, last = 0, t0 = performance.now();
    function size() {
      dpr = Math.min(devicePixelRatio || 1, 2);
      var r = canvas.getBoundingClientRect();
      W = Math.max(1, Math.round(r.width)); H = Math.max(1, Math.round(r.height));
      canvas.width = W * dpr; canvas.height = H * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    // isometric projection of a unit cube centred on the hero's upper half
    function P(x, y, z, s, cx, cy) {
      var ax = 0.86, ay = 0.5;
      return [cx + (x - z) * ax * s, cy + (x + z) * ay * s - y * s];
    }
    function plane(pts, fill, stroke) {
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]);
      for (var i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
      ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke();
    }
    function grid(a, b, c, d, n, stroke) { // lines across a quad a-b-c-d
      ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.beginPath();
      for (var i = 1; i < n; i++) {
        var f = i / n;
        ctx.moveTo(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f); ctx.lineTo(d[0] + (c[0] - d[0]) * f, d[1] + (c[1] - d[1]) * f);
        ctx.moveTo(a[0] + (d[0] - a[0]) * f, a[1] + (d[1] - a[1]) * f); ctx.lineTo(b[0] + (c[0] - b[0]) * f, b[1] + (c[1] - b[1]) * f);
      }
      ctx.stroke();
    }
    function frame(now) {
      raf = 0;
      if (!reduced) {
        if (now - last < 33) { raf = requestAnimationFrame(frame); return; } // ~30 fps is plenty for a drift
        last = now;
      }
      var t = reduced ? 0 : (now - t0) / 1000;
      ctx.clearRect(0, 0, W, H);
      // soft sage glow behind the headline
      var g = ctx.createRadialGradient(W * 0.5, H * 0.42, 0, W * 0.5, H * 0.42, Math.max(W, H) * 0.55);
      g.addColorStop(0, "rgba(82,184,155,0.16)"); g.addColorStop(0.5, "rgba(82,184,155,0.05)"); g.addColorStop(1, "rgba(7,8,12,0)");
      ctx.fillStyle = g; ctx.fillRect(0, 0, W, H);
      var s = Math.min(W, H) * 0.42, cx = W * 0.5, cy = H * 0.36;
      var sx = 0.5 + 0.28 * Math.sin(t * 0.21), sy = 0.5 + 0.28 * Math.sin(t * 0.17 + 1.7), sz = 0.5 + 0.28 * Math.sin(t * 0.13 + 3.1);
      var line = "rgba(236,238,244,0.07)", edge = "rgba(236,238,244,0.16)";
      // cube outline
      var c = [P(-.5,-.5,-.5,s,cx,cy),P(.5,-.5,-.5,s,cx,cy),P(.5,-.5,.5,s,cx,cy),P(-.5,-.5,.5,s,cx,cy),P(-.5,.5,-.5,s,cx,cy),P(.5,.5,-.5,s,cx,cy),P(.5,.5,.5,s,cx,cy),P(-.5,.5,.5,s,cx,cy)];
      ctx.strokeStyle = edge; ctx.lineWidth = 1; ctx.beginPath();
      [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]].forEach(function (e) { ctx.moveTo(c[e[0]][0], c[e[0]][1]); ctx.lineTo(c[e[1]][0], c[e[1]][1]); });
      ctx.stroke();
      // the three planes, painter's order by depth
      var y = sy - .5, x = sx - .5, z = sz - .5;
      var planes = [
        { d: y, pts: [P(-.5,y,-.5,s,cx,cy),P(.5,y,-.5,s,cx,cy),P(.5,y,.5,s,cx,cy),P(-.5,y,.5,s,cx,cy)], fill: "rgba(82,184,155,0.10)" },
        { d: x, pts: [P(x,-.5,-.5,s,cx,cy),P(x,-.5,.5,s,cx,cy),P(x,.5,.5,s,cx,cy),P(x,.5,-.5,s,cx,cy)], fill: "rgba(236,238,244,0.045)" },
        { d: z, pts: [P(-.5,-.5,z,s,cx,cy),P(.5,-.5,z,s,cx,cy),P(.5,.5,z,s,cx,cy),P(-.5,.5,z,s,cx,cy)], fill: "rgba(236,238,244,0.045)" }
      ];
      planes.sort(function (a, b) { return a.d - b.d; });
      planes.forEach(function (p) { plane(p.pts, p.fill, edge); grid(p.pts[0], p.pts[1], p.pts[2], p.pts[3], 8, line); });
      // a patch sweeping the axial plane: what the engine reads, region by region
      var k = (t * 0.12) % 1, px = -0.5 + (k * 4 % 1) * 0.75, pz = -0.5 + Math.floor(k * 4) * 0.25;
      plane([P(px,y,pz,s,cx,cy),P(px+.25,y,pz,s,cx,cy),P(px+.25,y,pz+.25,s,cx,cy),P(px,y,pz+.25,s,cx,cy)], "rgba(82,184,155,0.28)", "rgba(143,224,198,0.7)");
      // fade to the page below
      var f = ctx.createLinearGradient(0, H * 0.6, 0, H);
      f.addColorStop(0, "rgba(7,8,12,0)"); f.addColorStop(1, "rgba(7,8,12,1)");
      ctx.fillStyle = f; ctx.fillRect(0, 0, W, H);
      if (!reduced && !document.hidden) raf = requestAnimationFrame(frame);
    }
    function start() { if (!raf) raf = requestAnimationFrame(frame); }
    size(); start();
    addEventListener("resize", function () { size(); if (reduced) frame(performance.now()); });
    document.addEventListener("visibilitychange", function () { if (!document.hidden) start(); });
  }

  // ---- search: the site's pages, docs, tools and apps, filtered as you type ----
  var INDEX = [{"k":"page","t":"KonfAI","d":"KonfAI: a declarative medical-imaging execution engine, its published apps, its MCP server, Scout and Studio.","u":"index.html"},{"k":"page","t":"KonfAI Apps","d":"Published KonfAI applications: segmentation, synthesis and registration systems packaged with one contract.","u":"apps.html"},{"k":"page","t":"KonfAI MCP","d":"The MCP server that lets an agent drive KonfAI: fifty tools, one per object.","u":"mcp.html"},{"k":"page","t":"KonfAI Scout","d":"An agent that inspects a segmentation, registration or synthesis result and reports localised defects with evidence.","u":"scout.html"},{"k":"page","t":"KonfAI Studio","d":"A chat web UI over the KonfAI MCP server: inspect, train, predict, evaluate, compare and view from one conversation.","u":"studio.html"},{"k":"docs","t":"Agents and MCP","d":"What the MCP server lets an agent do.","u":"docs/agents-mcp.html"},{"k":"docs","t":"Apps","d":"Published bundles, Hugging Face, the one-line CLI, 3D Slicer.","u":"docs/apps.html"},{"k":"docs","t":"Configuration","d":"One YAML per workflow: roots, dataset, groups, transforms, patches, models.","u":"docs/configuration.html"},{"k":"docs","t":"Datasets","d":"The layout KonfAI reads and the formats it reads regionally.","u":"docs/datasets.html"},{"k":"docs","t":"Edge","d":"KonfAI apps in a browser tab, no server.","u":"docs/edge.html"},{"k":"docs","t":"Get started","d":"Install KonfAI and run the Segmentation example.","u":"docs/get-started.html"},{"k":"docs","t":"Documentation","d":"What these pages cover.","u":"docs/index.html"},{"k":"docs","t":"Scout","d":"A visual inspector of results.","u":"docs/scout.html"},{"k":"docs","t":"Studio","d":"A chat web UI over the MCP server.","u":"docs/studio.html"},{"k":"docs","t":"Workflows","d":"TRAIN, RESUME, PREDICTION, EVALUATION, TRANSFORM and the options that matter.","u":"docs/workflows.html"},{"k":"tool","t":"describe_konfai_capabilities","d":"What KonfAI can do and which tool to reach for: the four workflows, the component kinds, the extension model, which actions are safe and whi","u":"mcp.html#discovery"},{"k":"tool","t":"list_components","d":"Built-in KonfAI components of one kind (model, loss, metric, transform, augmentation, scheduler, ...): name, where it goes in the YAML, the ","u":"mcp.html#discovery"},{"k":"tool","t":"inspect_object_signature","d":"Signature, defaults, doc and contract hints of any configurable classpath: built-in, local File:Class, external package.module:Class, or a d","u":"mcp.html#discovery"},{"k":"tool","t":"describe_config_schema","d":"Top-level schema of a workflow config (train, prediction, evaluation, transform), generated from KonfAI's own classes: each field's literal ","u":"mcp.html#discovery"},{"k":"tool","t":"describe_extension_points","d":"How to add a component of a kind (loss, metric, model, transform, augmentation, scheduler, pretrained model): the base class, the required m","u":"mcp.html#discovery"},{"k":"tool","t":"check_component","d":"Pre-flight a component before wiring it into a config.","u":"mcp.html#discovery"},{"k":"tool","t":"inspect_dataset","d":"Groups, entry hints, ambiguities and sampled statistics (shapes channel-first) of one dataset root (include_stats=False for a fast structure","u":"mcp.html#dataset-onboarding"},{"k":"tool","t":"read_file","d":"Read a bounded text range of one file.","u":"mcp.html#dataset-onboarding"},{"k":"tool","t":"preview_volume","d":"One slice of a volume as a PNG (any SimpleITK-readable file), windowed between the 1st and 99th percentile and downsampled, for a visual che","u":"mcp.html#dataset-onboarding"},{"k":"tool","t":"prepare_dataset_aliases","d":"Copy, symlink or move dataset files under new group names when the content is right but the filenames do not match the intended config.","u":"mcp.html#dataset-onboarding"},{"k":"tool","t":"design_config_strategy","d":"A config-writing plan from the task, the dataset root(s), the group roles and the modeling intent: dataset summary, config plan, customisati","u":"mcp.html#config-authoring"},{"k":"tool","t":"initialize_session","d":"Create or reset the current session workspace, optionally seeded from an example template (read_file('templates://list') lists them; include","u":"mcp.html#config-authoring"},{"k":"tool","t":"write_workflow_config","d":"Write one workflow YAML (train, prediction, evaluation, transform) into the session and check it against KonfAI's schema: schema_errors list","u":"mcp.html#config-authoring"},{"k":"tool","t":"write_session_file","d":"Write one support file into the current session (Model.py, Loss.py, a transform, a manifest).","u":"mcp.html#config-authoring"},{"k":"tool","t":"check_config","d":"Check the session's workflow config.","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"summarize_session","d":"One compact snapshot of the current session: readiness, jobs, metric summaries, and on request the leaderboard, a log tail and a config vali","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"leaderboard","d":"Ranked metrics across the session's completed runs, one metric and split, direction-aware, with the available metrics and splits.","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"get_run_metrics","d":"Full evaluation metrics (per-case values and aggregates) of one run or app trial, from Evaluations/<run>/Metric_<split>.json.","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"compare_runs","d":"Two runs metric by metric on aligned cases: means, per-case deltas, a direction-aware winner (loss-like metrics count lower as better).","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"diff_run_configs","d":"Unified diff of the exact configs two jobs ran with, from their immutable launch-time snapshots.","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"describe_model_outputs","d":"The addressable module paths of the session config's networks: the exact keys outputs_criterions and outputs_dataset accept (terminal=true m","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"read_training_curves","d":"A run's training curves (scalars over iterations) from its TensorBoard event files, downsampled to max_points.","u":"mcp.html#validation-and-iteration"},{"k":"tool","t":"run_train","d":"Launch a training job from the session Config.yml (after check_config).","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"run_resume","d":"Resume a training from a checkpoint (model, optimizer, scheduler and counters restored), or warm-start a fine-tune with weights_only=True (w","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"run_prediction","d":"Launch a prediction job from the session Prediction.yml with a checkpoint of the session (models picks which).","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"run_evaluation","d":"Launch an evaluation job from the session Evaluation.yml on existing predictions.","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"run_transform","d":"Batch-process a dataset with the session Transform.yml chain: read, transform, write.","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"run_batch","d":"Run several training configs sequentially server-side (a sweep, fold training), blocking until the batch ends, with per-run outcomes.","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"generate_folds","d":"Split a dataset into K cross-validation folds: one case-list file per fold in the session, plus the subset stanzas to paste into each config","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"get_job_status","d":"The state of one job (job_id) or of every job of the session (no job_id).","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"cancel_job","d":"Stop a running job: requests cancellation and waits briefly for a clean shutdown.","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"control_training","d":"Steer a RUNNING training without restarting it: lr and it_validation are applied at its next poll boundary and recorded in the run's config ","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"read_live_metrics","d":"Parsed runtime metrics of a running or finished training (latest snapshot, recent entries, per-stage summaries) instead of raw log lines.","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"read_job_log","d":"A job's console or runtime log: the last max_lines, or the lines matching a regex, to find the traceback.","u":"mcp.html#execution-and-monitoring"},{"k":"tool","t":"list_apps","d":"Published KonfAI apps from the catalogue (shipped, workspace file, KONFAI_MCP_APP_CATALOG), HuggingFace repos expanded.","u":"mcp.html#apps"},{"k":"tool","t":"describe_app","d":"One app's manifest: description, input and output modalities, inference, evaluation and uncertainty capabilities, checkpoints, terminology.","u":"mcp.html#apps"},{"k":"tool","t":"run_app","d":"Run a published app as published on the user's data: action infer, evaluate (the app's own Evaluation.yml against gt), uncertainty, or pipel","u":"mcp.html#apps"},{"k":"tool","t":"fine_tune_app","d":"Fine-tune a published app's checkpoint on the user's dataset without authoring a config (epochs, it_validation, lr, batch_size as parameters","u":"mcp.html#apps"},{"k":"tool","t":"fetch_app","d":"Bring a published app into the session.","u":"mcp.html#apps"},{"k":"tool","t":"package_app_from_session","d":"Package a model trained in this session into a runnable app bundle (app.json, config, checkpoint, optional Model.py, requirements, support_f","u":"mcp.html#apps"},{"k":"tool","t":"app_sources","d":"Edit the workspace app catalogue that list_apps merges: action 'add' persists an app path or HuggingFace repo across sessions, 'remove' drop","u":"mcp.html#apps"},{"k":"tool","t":"run_registration_evaluate","d":"Score a registration against what the pair has: labels (Dice), landmarks (TRE in mm, fixed points pushed through the transform), same-modali","u":"mcp.html#apps"},{"k":"tool","t":"run_scout_inspection","d":"Have KonfAI Scout inspect one result visually (segmentation, registration or synthesis): a vision model looks at slices, zooms and overlays,","u":"mcp.html#visual-inspection"},{"k":"tool","t":"read_scout_report","d":"A finished Scout inspection by name: verdict, observations (kind, defect, severity, confidence, box in mm, cited views and measurements), me","u":"mcp.html#visual-inspection"},{"k":"tool","t":"fidopt_catalog","d":"FidOpt's benchmark catalogue through its own code: one line per dataset (task, anatomy, modalities, cases; task= and anatomy= narrow it) and","u":"mcp.html#fidopt"},{"k":"tool","t":"fidopt_table","d":"A FidOpt dataset's board (fidopt table --json): one line per method and machine, published scores and local rows.","u":"mcp.html#fidopt"},{"k":"tool","t":"fidopt_run","d":"Launch one FidOpt operation as a tracked job: check, run or tune a method on a dataset's cases (only the operations KONFAI_MCP_FIDOPT_ALLOWE","u":"mcp.html#fidopt"},{"k":"tool","t":"session","d":"Manage session workspaces: action 'list' (says whether a host pinned the session); 'create' (name; switches to it; from_example seeds it in ","u":"mcp.html#session-lifecycle"},{"k":"tool","t":"export_run_record","d":"The reproducibility record of one run in one payload: the job manifest (command, devices, environment with package versions), the launch-tim","u":"mcp.html#session-lifecycle"},{"k":"tool","t":"delete_run","d":"Delete one run's outputs of a kind (train: its Statistics and Checkpoints; prediction, evaluation, uncertainty: its folder; transform: its p","u":"mcp.html#session-lifecycle"},{"k":"app","t":"TotalSegmentator-KonfAI","d":"whole-body CT or MRI segmentation with TotalSegmentator models; CT: 117 labels, 5 models; MRI: 50 labels, 2 models","u":"apps.html"},{"k":"app","t":"MRSegmentator-KonfAI","d":"whole-body MRI segmentation with MRSegmentator models; 40 labels, 5-fold ensemble","u":"apps.html"},{"k":"app","t":"ImpactSynth","d":"synthetic CT generation with IMPACT-Synth models; three MR and CBCT to sCT variants, 2.5D UNet++, 5 models each","u":"apps.html"},{"k":"app","t":"ImpactSeg","d":"multimodal anatomical segmentation with IMPACT-Seg models; one model segments 11 structures from CT, MRI or CBCT","u":"apps.html"},{"k":"app","t":"IMPACT-Reg","d":"multimodal medical image registration with IMPACT-Reg presets; 13 presets across elastix with IMPACT, ConvexAdam and FireANTs","u":"apps.html"}];
  var input = document.getElementById("q"), box = document.getElementById("results");
  if (input && box) {
    var labels = { page: "page", docs: "docs", tool: "tool", app: "app" };
    var sel = -1, hits = [];
    function render(q) {
      q = q.trim().toLowerCase();
      box.innerHTML = ""; sel = -1; hits = [];
      if (!q) { box.classList.remove("on"); return; }
      hits = INDEX.filter(function (e) { return (e.t + " " + e.d).toLowerCase().indexOf(q) >= 0; }).slice(0, 8);
      if (!hits.length) { box.innerHTML = '<div class="none">Nothing matches.</div>'; box.classList.add("on"); return; }
      hits.forEach(function (e, i) {
        var a = document.createElement("a"); a.href = e.u; a.setAttribute("role", "option"); a.id = "hit-" + i;
        var k = document.createElement("span"); k.className = "k"; k.textContent = labels[e.k] || e.k;
        var t = document.createElement("span"); t.className = "t"; t.textContent = e.t;
        var d = document.createElement("span"); d.className = "d"; d.textContent = e.d;
        t.appendChild(d); a.appendChild(k); a.appendChild(t); box.appendChild(a);
      });
      box.classList.add("on");
    }
    function mark() { Array.prototype.forEach.call(box.children, function (c, i) { c.setAttribute("aria-selected", i === sel ? "true" : "false"); }); }
    input.addEventListener("input", function () { render(input.value); });
    input.addEventListener("keydown", function (ev) {
      if (ev.key === "Escape") { box.classList.remove("on"); input.blur(); }
      else if (ev.key === "ArrowDown" && hits.length) { ev.preventDefault(); sel = (sel + 1) % hits.length; mark(); }
      else if (ev.key === "ArrowUp" && hits.length) { ev.preventDefault(); sel = (sel - 1 + hits.length) % hits.length; mark(); }
      else if (ev.key === "Enter" && hits.length) { location.href = hits[Math.max(sel, 0)].u; }
    });
    document.addEventListener("click", function (ev) { if (!box.contains(ev.target) && ev.target !== input) box.classList.remove("on"); });
  }

  // ---- phone menu ----
  var bar = document.querySelector(".bar"), menu = document.getElementById("menu");
  if (bar && menu) menu.addEventListener("click", function () {
    var open = bar.classList.toggle("open"); menu.setAttribute("aria-expanded", open ? "true" : "false");
  });
})();
