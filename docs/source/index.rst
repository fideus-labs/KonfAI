KonfAI
======

.. raw:: html

   <div class="konfai-landing">
     <section class="konfai-hero">
       <p class="konfai-eyebrow">YAML-driven deep learning for medical imaging &middot; built on PyTorch</p>
       <p class="konfai-tagline">
         Describe an entire pipeline &mdash; data, model, losses, metrics,
         augmentations, and the train / predict / evaluate workflow &mdash; in
         <strong>configuration</strong>, not orchestration scripts. The config
         <em>is</em> the experiment: reproducible, inspectable, shareable.
       </p>
       <div class="konfai-cta">
         <a class="konfai-btn konfai-btn-primary" href="quickstart.html">Get started &rarr;</a>
         <a class="konfai-btn konfai-btn-ghost" href="reference/components/index.html">Browse components</a>
         <a class="konfai-btn konfai-btn-ghost" href="https://github.com/vboussot/KonfAI">GitHub</a>
       </div>
     </section>

     <section class="konfai-cards">
       <a class="konfai-card" href="quickstart.html">
         <span class="konfai-card-ico">🚀</span>
         <span class="konfai-card-title">Quickstart</span>
         <span class="konfai-card-desc">Install, train, predict and evaluate the demo in one sitting.</span>
       </a>
       <a class="konfai-card" href="concepts/index.html">
         <span class="konfai-card-ico">🧩</span>
         <span class="konfai-card-title">Core concepts</span>
         <span class="konfai-card-desc">How YAML becomes Python objects, and the patch-based data model.</span>
       </a>
       <a class="konfai-card" href="reference/components/index.html">
         <span class="konfai-card-ico">📚</span>
         <span class="konfai-card-title">Component catalogue</span>
         <span class="konfai-card-desc">Every model, loss, metric, transform and backend you can name in YAML.</span>
       </a>
       <a class="konfai-card" href="examples/index.html">
         <span class="konfai-card-ico">🧪</span>
         <span class="konfai-card-title">Examples</span>
         <span class="konfai-card-desc">Runnable Segmentation &amp; Synthesis workflows to copy and adapt.</span>
       </a>
       <a class="konfai-card" href="concepts/apps.html">
         <span class="konfai-card-ico">📦</span>
         <span class="konfai-card-title">Apps &amp; API</span>
         <span class="konfai-card-desc">Ship a workflow behind a CLI, an HTTP server, or the Python API.</span>
       </a>
       <a class="konfai-card" href="ecosystem/index.html">
         <span class="konfai-card-ico">🗺️</span>
         <span class="konfai-card-title">Ecosystem</span>
         <span class="konfai-card-desc">konfai-apps, SlicerKonfAI, KonfAI-MCP &mdash; what is shipped.</span>
       </a>
     </section>

     <figure class="konfai-arch">
       <svg viewBox="0 0 864 196" role="img" aria-label="KonfAI architecture: one YAML config is built into an object graph by reflection and run as train, prediction or evaluation workflows." preserveAspectRatio="xMidYMid meet">
         <defs>
           <marker id="kf-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
             <path d="M0,0 L10,5 L0,10 z" fill="#14b8a6"/>
           </marker>
           <style>
             .k-t{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif}
             .k-lbl{fill:#14b8a6;font-weight:700;font-size:12px;letter-spacing:.09em;text-transform:uppercase}
             .k-card{fill:none;stroke:currentColor;stroke-opacity:.22;stroke-width:1.4}
             .k-code{fill:currentColor;opacity:.6;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:12px}
             .k-chip{fill:#14b8a6;fill-opacity:.10;stroke:#14b8a6;stroke-opacity:.45;stroke-width:1.3}
             .k-chip-t{fill:currentColor;opacity:.88;font-weight:600;font-size:13px}
             .k-pill{fill:currentColor;fill-opacity:.04;stroke:currentColor;stroke-opacity:.3;stroke-width:1.3}
             .k-pill-t{fill:currentColor;font-weight:700;font-size:13px;letter-spacing:.03em}
             .k-arrow{stroke:#14b8a6;stroke-width:2}
           </style>
         </defs>

         <!-- Stage 1: YAML config -->
         <text class="k-t k-lbl" x="106" y="22" text-anchor="middle">YAML config</text>
         <rect class="k-card" x="8" y="34" width="196" height="120" rx="12"/>
         <text class="k-t k-code" x="26" y="66">Trainer:</text>
         <text class="k-t k-code" x="26" y="90">  Model: UNet.yml</text>
         <text class="k-t k-code" x="26" y="114">  Dataset: {…}</text>
         <text class="k-t k-code" x="26" y="138">  epochs: 100</text>

         <line class="k-arrow" x1="210" y1="94" x2="250" y2="94" marker-end="url(#kf-arrow)"/>

         <!-- Stage 2: config-by-reflection -->
         <text class="k-t k-lbl" x="454" y="22" text-anchor="middle">config-by-reflection</text>
         <rect class="k-card" x="258" y="34" width="392" height="140" rx="12"/>
         <rect class="k-chip" x="276" y="52" width="172" height="46" rx="9"/>
         <text class="k-t k-chip-t" x="362" y="80" text-anchor="middle">Model</text>
         <rect class="k-chip" x="460" y="52" width="172" height="46" rx="9"/>
         <text class="k-t k-chip-t" x="546" y="80" text-anchor="middle">Data · patches</text>
         <rect class="k-chip" x="276" y="110" width="172" height="46" rx="9"/>
         <text class="k-t k-chip-t" x="362" y="138" text-anchor="middle">Losses · Metrics</text>
         <rect class="k-chip" x="460" y="110" width="172" height="46" rx="9"/>
         <text class="k-t k-chip-t" x="546" y="138" text-anchor="middle">Optimizer · LR</text>

         <line class="k-arrow" x1="656" y1="94" x2="696" y2="94" marker-end="url(#kf-arrow)"/>

         <!-- Stage 3: workflows -->
         <text class="k-t k-lbl" x="780" y="22" text-anchor="middle">workflows</text>
         <rect class="k-pill" x="704" y="42" width="152" height="34" rx="17"/>
         <text class="k-t k-pill-t" x="780" y="64" text-anchor="middle">TRAIN</text>
         <rect class="k-pill" x="704" y="90" width="152" height="34" rx="17"/>
         <text class="k-t k-pill-t" x="780" y="112" text-anchor="middle">PREDICTION</text>
         <rect class="k-pill" x="704" y="138" width="152" height="34" rx="17"/>
         <text class="k-t k-pill-t" x="780" y="160" text-anchor="middle">EVALUATION</text>
       </svg>
       <figcaption>
         KonfAI reads one YAML file, builds the object graph by reflection, and runs it as a
         reproducible workflow &mdash; writing checkpoints, predictions and evaluations to a workspace.
       </figcaption>
     </figure>
   </div>

.. note::

   New here? The fastest path is :doc:`quickstart`, then copy
   ``examples/Segmentation`` and run ``konfai TRAIN``. Come back to
   :doc:`concepts/index` when you want to adapt the YAML, and to
   :doc:`reference/components/index` to see what ships in the box.

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
   contributing
   development
   architecture
