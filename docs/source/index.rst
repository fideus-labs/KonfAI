KonfAI
======

.. raw:: html

   <div class="kf-landing">

     <!-- ======================= HERO ======================= -->
     <section class="kf-hero">
       <div class="kf-hero-grid">
         <div>
           <p class="kf-eyebrow">YAML-driven deep learning &middot; medical imaging &middot; PyTorch</p>
           <h2 class="kf-title" style="border:0; padding:0;">The config <em>is</em> the experiment.</h2>
           <p class="kf-lede">
             Describe the whole pipeline &mdash; data, model, losses, metrics, augmentations, and the
             train&nbsp;/&nbsp;predict&nbsp;/&nbsp;evaluate workflow &mdash; in configuration, not orchestration
             scripts. KonfAI builds Python objects from that YAML by reflection, and writes the
             fully-resolved config back to disk. Reproducible, inspectable, shareable.
           </p>
           <div class="kf-cta">
             <a class="kf-btn kf-btn-primary" href="quickstart.html">Run it in 5 minutes &rarr;</a>
             <a class="kf-btn kf-btn-ghost" href="#mental-model">See the mental model</a>
             <a class="kf-btn kf-btn-ghost" href="https://github.com/vboussot/KonfAI">GitHub</a>
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
           <p>Volumes are never loaded whole. Data is read as overlapping patches and predictions
              are reassembled with overlap blending &mdash; large 3D scans on modest hardware.</p>
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
         <a class="kf-nextcard kf-h-coral" href="usage/apps.html">
           <span class="intent">Ship</span>
           <h3>As an app</h3>
           <p>Package a stable workflow behind <code>konfai-apps</code> &mdash; local, HuggingFace,
              or an HTTP server.</p>
           <span class="go">Apps &amp; API &rarr;</span>
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
           <a class="kf-h-steel" href="reference/python-api.html">Python API</a>
           <a class="kf-h-amber" href="troubleshooting.html">Troubleshooting</a>
         </div>
       </div>

       <div class="kf-proof">
         <span class="label">Proven at MICCAI</span>
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
   :caption: Core concepts
   :hidden:

   concepts/index
   config_guide/index

.. toctree::
   :maxdepth: 2
   :caption: Guides
   :hidden:

   usage/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   reference/index

.. toctree::
   :maxdepth: 2
   :caption: Project
   :hidden:

   examples/index
   ecosystem/index
   troubleshooting
   development
