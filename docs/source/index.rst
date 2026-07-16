KonfAI
======

.. raw:: html

   <div class="kf-landing">

     <!-- ======================= HERO ======================= -->
     <section class="kf-hero">
       <div class="kf-hero-grid">
         <div>
           <p class="kf-eyebrow">Scale &middot; reproduce &middot; ship and automate</p>
           <h2 class="kf-title" style="border:0; padding:0;">From medical-image storage to a reusable application.</h2>
           <p class="kf-lede">
             KonfAI is a declarative medical-imaging execution engine. It connects regional data access,
             patch execution, inspectable PyTorch graphs, training, prediction, evaluation, and medical-image
             outputs through one reproducible workflow &mdash; then packages it for people, services, or agents.
           </p>
           <div class="kf-cta">
             <a class="kf-btn kf-btn-primary" href="quickstart.html">Run your first workflow &rarr;</a>
             <a class="kf-btn kf-btn-ghost" href="examples/visual-gallery.html">See it on real data</a>
             <a class="kf-btn kf-btn-ghost" href="https://github.com/fideus-labs/KonfAI">GitHub</a>
           </div>
           <div class="kf-herometa">
             <span><b>pip</b> install "konfai[imaging]"</span>
             <span><b>Apache-2.0</b></span>
             <span><b>Python</b> 3.10+</span>
           </div>
         </div>

         <div>
           <div class="kf-codecard">
             <div class="kf-chead"><span class="kf-dots"><i></i><i></i><i></i></span><span>Config.yml</span></div>
             <pre><span class="k">Trainer:</span>
     <span class="k">train_name:</span> <span class="s">SEG_BASELINE</span>
     <span class="k">Model:</span>
       <span class="k">classpath:</span> <span class="s">UNet.yml</span>   <span class="c"># routed graph</span>
     <span class="k">Dataset:</span>
       <span class="k">dataset_filenames:</span> <span class="s">[ ./Dataset:mha ]</span>
       <span class="k">groups_src:</span> <span class="s">{ CT: {...}, SEG: {...} }</span>
     <span class="k">epochs:</span> <span class="s">100</span></pre>
           </div>
           <div class="kf-flowdown"><span>reflection builds the object graph</span></div>
           <div class="kf-codecard">
             <div class="kf-chead"><span class="kf-dots"><i></i><i></i><i></i></span><span>terminal</span></div>
             <pre><span class="p">$</span> <span class="cmd">konfai TRAIN</span>      <span class="fl">-y --gpu 0 --config Config.yml</span>
   <span class="p">$</span> <span class="cmd">konfai PREDICTION</span> <span class="fl">--config Prediction.yml --models &hellip;/best.pt</span>
   <span class="p">$</span> <span class="cmd">konfai EVALUATION</span> <span class="fl">--config Evaluation.yml</span></pre>
           </div>
         </div>
       </div>
     </section>

     <!-- ================== PROOF STRIP ===================== -->
     <section class="kf-proofstrip" aria-label="Real medical-imaging results">
       <a class="kf-proofcard" href="examples/visual-gallery.html">
         <img src="_static/apps/impact-synth/totalsegmentator.png" alt="Whole-body CT segmentation overlay">
         <span><b>Whole-body segmentation</b>CT structures, from a packaged model</span>
       </a>
       <a class="kf-proofcard" href="examples/visual-gallery.html">
         <img src="_static/apps/impact-synth/synthetic-ct.png" alt="MR-to-synthetic-CT result">
         <span><b>MR &rarr; synthetic CT</b>continuous Hounsfield output</span>
       </a>
       <a class="kf-proofcard" href="usage/large-images.html">
         <img src="_static/gallery/scale-omezarr.webp" alt="Streaming patches from an OME-Zarr store">
         <span><b>Scale from storage, not RAM</b>stream patches from OME-Zarr</span>
       </a>
     </section>

     <!-- ================== MENTAL MODEL ==================== -->
     <section class="kf-block" id="mental-model">
       <div class="kf-sechead">
         <p class="kf-eyebrow">The 20-second mental model</p>
         <h2 style="border:0; padding:0;">One config, built once, run three ways.</h2>
         <p>Three YAML files, one root key each, three commands. Each command reads <em>its</em> file,
            builds the object graph by reflection, and writes its outputs. That's the whole mapping
            to remember:</p>
       </div>

       <div class="kf-lanes">
         <div class="kf-lrow kf-lhead">
           <span class="c1">Your YAML &middot; root key</span>
           <span></span>
           <span class="c3">One command</span>
           <span></span>
           <span class="c5">Writes to the workspace</span>
         </div>
         <div class="kf-lrow">
           <div class="kf-lfile"><span class="fname">Config.yml</span><span class="rootkey">Trainer:</span></div>
           <span class="kf-larrow">&rarr;</span>
           <span class="kf-lcmd"><span class="p">$</span> konfai TRAIN</span>
           <span class="kf-larrow">&rarr;</span>
           <div class="kf-lout"><b>Checkpoints/</b>&lt;train_name&gt;/ &middot; <b>Statistics/</b>&lt;train_name&gt;/</div>
         </div>
         <div class="kf-lrow">
           <div class="kf-lfile"><span class="fname">Prediction.yml</span><span class="rootkey">Predictor:</span></div>
           <span class="kf-larrow">&rarr;</span>
           <span class="kf-lcmd"><span class="p">$</span> konfai PREDICTION</span>
           <span class="kf-larrow">&rarr;</span>
           <div class="kf-lout"><b>Predictions/</b>&lt;train_name&gt;/</div>
         </div>
         <div class="kf-lrow">
           <div class="kf-lfile"><span class="fname">Evaluation.yml</span><span class="rootkey">Evaluator:</span></div>
           <span class="kf-larrow">&rarr;</span>
           <span class="kf-lcmd"><span class="p">$</span> konfai EVALUATION</span>
           <span class="kf-larrow">&rarr;</span>
           <div class="kf-lout"><b>Evaluations/</b>&lt;train_name&gt;/Metric_TRAIN.json</div>
         </div>
         <div class="kf-lfoot">
           <span>Same engine underneath: reflection reads the root key and builds
                 <code>Model &middot; Dataset &middot; Losses &middot; Optimizer</code>.</span>
           <span>Every output folder is keyed by <code>train_name</code> &mdash; keep it consistent
                 across the three files.</span>
         </div>
       </div>

       <div class="kf-pillars">
         <div class="kf-pillar kf-h-teal">
           <span class="tag">Reflection</span>
           <h3>Config by reflection</h3>
           <p>A callable's signature is read and its arguments are built from the YAML it owns
              &mdash; recursively. Reading a config resolves and rewrites it, so a run leaves a
              complete record.</p>
         </div>
         <div class="kf-pillar kf-h-steel">
           <span class="tag">Imaging</span>
           <h3>Lazy, patch-based</h3>
           <p>Compatible chains stream only the source region required by each patch; other chains
              use a bounded case buffer. Predictions are reassembled with overlap blending.</p>
         </div>
         <div class="kf-pillar kf-h-violet">
           <span class="tag">Models</span>
           <h3>Declarative graphs</h3>
           <p>Networks are routed <code>add_module</code> graphs &mdash; a Python class, or an
              entire model written as a <code>.yml</code>. Named outputs are addressable from YAML.</p>
         </div>
       </div>
     </section>

     <!-- ================== WHERE TO GO ===================== -->
     <section class="kf-block">
       <div class="kf-sechead">
         <p class="kf-eyebrow">Where to go next</p>
         <h2 style="border:0; padding:0;">Pick the path that matches your goal.</h2>
       </div>

       <div class="kf-nextgrid">
         <a class="kf-nextcard kf-h-teal" href="quickstart.html">
           <span class="intent">Start</span>
           <h3>Your first run</h3>
           <p>Install, download the demo dataset, then train, predict and evaluate the shipped
              segmentation baseline &mdash; step by step.</p>
           <span class="go">Quickstart &rarr;</span>
         </a>
         <a class="kf-nextcard kf-h-violet" href="concepts/index.html">
           <span class="intent">Understand</span>
           <h3>The config model</h3>
           <p>How YAML becomes Python objects, classpaths, named module outputs, and the
              rewrite-on-read behaviour.</p>
           <span class="go">Core concepts &rarr;</span>
         </a>
         <a class="kf-nextcard kf-h-steel" href="usage/large-images.html">
           <span class="intent">Scale</span>
           <h3>Process large images</h3>
           <p>Choose cache, bounded buffering, or direct regional reads and understand exactly when streaming falls back.</p>
           <span class="go">Large-image guide &rarr;</span>
         </a>
         <a class="kf-nextcard kf-h-teal" href="usage/adopting-konfai.html">
           <span class="intent">Reuse</span>
           <h3>Bring PyTorch or MONAI</h3>
           <p>Adopt gradually, reuse existing components and weights, and choose fairly between KonfAI and neighbouring tools.</p>
           <span class="go">Adoption guide &rarr;</span>
         </a>
         <a class="kf-nextcard kf-h-coral" href="usage/apps.html">
           <span class="intent">Ship</span>
           <h3>As an app</h3>
           <p>Package a stable workflow behind <code>konfai-apps</code> &mdash; local, HuggingFace,
              or an HTTP server.</p>
           <span class="go">Apps &amp; API &rarr;</span>
         </a>
         <a class="kf-nextcard kf-h-amber" href="usage/mcp.html">
           <span class="intent">Automate</span>
           <h3>Drive it with an agent</h3>
           <p>An MCP server lets an LLM inspect data, write &amp; validate configs, run
              train/predict/evaluate &mdash; and <em>use, fine-tune, or package apps</em>.</p>
           <span class="go">Agents &amp; MCP &rarr;</span>
         </a>
       </div>

       <div class="kf-docindex">
         <div class="dhead">
           <h3>Or browse the full documentation</h3>
           <span>Every page stays one click away &mdash; this landing is a map, not a wall.</span>
         </div>
         <div class="kf-doclinks">
           <a class="kf-h-violet" href="concepts/index.html">Core concepts</a>
           <a class="kf-h-teal" href="config_guide/index.html">Config guide</a>
           <a class="kf-h-teal" href="reference/components/index.html">Component catalogue</a>
           <a class="kf-h-steel" href="reference/cli.html">CLI reference</a>
           <a class="kf-h-violet" href="examples/index.html">Examples</a>
           <a class="kf-h-coral" href="usage/apps.html">Apps &amp; API server</a>
           <a class="kf-h-amber" href="usage/mcp.html">Agents &amp; MCP</a>
           <a class="kf-h-steel" href="reference/python-api.html">Python API</a>
           <a class="kf-h-amber" href="troubleshooting.html">Troubleshooting</a>
         </div>
       </div>

       <div class="kf-proof">
         <span class="label">Top-ranking at recent MICCAI challenges</span>
         <div class="chips">
           <span class="chip kf-h-teal"><b>SynthRAD</b></span>
           <span class="chip kf-h-steel"><b>TrackRAD</b></span>
           <span class="chip kf-h-violet"><b>CURVAS</b></span>
           <span class="chip kf-h-coral"><b>PANTHER</b></span>
         </div>
         <span class="label label-right">segmentation &middot; registration &middot; synthesis</span>
       </div>
     </section>

   </div>

.. toctree::
   :maxdepth: 2
   :caption: Getting started
   :hidden:

   getting-started/installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Task-oriented guides
   :hidden:

   usage/index

.. toctree::
   :maxdepth: 2
   :caption: Core concepts
   :hidden:

   concepts/index
   config_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Apps and automation
   :hidden:

   usage/apps
   usage/mcp
   ecosystem/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Project
   :hidden:

   troubleshooting
   development
