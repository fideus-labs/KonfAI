KonfAI documentation
====================

.. raw:: html

   <div class="kf-landing kf-start">
     <div class="kf-start-hero">
       <div class="kf-start-intro">
         <p class="kf-eyebrow">KonfAI documentation</p>
         <h2 class="kf-start-title">Run a model.<br><em>Build a workflow.</em></h2>
         <p class="kf-lede">Medical-image segmentation, synthesis and registration. Start with a published app or build a reproducible workflow around your own model.</p>
         <div class="kf-cta">
           <a class="kf-btn kf-btn-primary" href="usage/apps.html">Run your first app <span aria-hidden="true">→</span></a>
           <a class="kf-btn kf-btn-ghost" href="getting-started/installation.html">Installation</a>
         </div>
       </div>
       <figure class="kf-start-preview">
         <a href="examples/visual-gallery.html" aria-label="View real outputs and their data sources in the gallery">
           <span><img src="_static/gallery/capabilities/seg-input.png" alt="Axial pelvis CT slice from the demo dataset." width="460" height="460" decoding="async"><b>CT input</b></span>
           <span><img src="_static/gallery/capabilities/seg-output.png" alt="The same CT slice with the ImpactSeg anatomy labels overlaid." width="460" height="460" decoding="async"><b>Segmentation</b></span>
         </a>
         <figcaption>Real data, real outputs. <a href="examples/visual-gallery.html">Explore the gallery <span aria-hidden="true">↗</span></a></figcaption>
       </figure>
     </div>
     <section class="kf-start-tasks">
       <h2>Choose your task</h2>
       <ul class="kf-nextgrid" role="list">
         <li><a class="kf-nextcard kf-h-teal" href="usage/apps.html" aria-labelledby="task-apps">
           <span class="intent">Published models</span>
           <h3 id="task-apps">Run a trained model</h3>
           <p>Segment an image or generate a synthetic CT with a ready-to-use app.</p>
           <span class="go">Run your first app <span aria-hidden="true">→</span></span>
         </a></li>
         <li><a class="kf-nextcard kf-h-steel" href="quickstart.html" aria-labelledby="task-train">
           <span class="intent">First experiment</span>
           <h3 id="task-train">Train, predict, evaluate</h3>
           <p>Complete a small CPU experiment with generated data and verified outputs.</p>
           <span class="go">Follow the quickstart <span aria-hidden="true">→</span></span>
         </a></li>
         <li><a class="kf-nextcard kf-h-violet" href="usage/adopting-konfai.html" aria-labelledby="task-model">
           <span class="intent">Your own model</span>
           <h3 id="task-model">Bring an existing model</h3>
           <p>Use a PyTorch or MONAI model, or transfer compatible pretrained weights.</p>
           <span class="go">Adopt KonfAI <span aria-hidden="true">→</span></span>
         </a></li>
         <li><a class="kf-nextcard kf-h-teal" href="usage/making-data.html" aria-labelledby="task-data">
           <span class="intent">Data preparation</span>
           <h3 id="task-data">Prepare a dataset</h3>
           <p>Resample, transform or combine volumes before your next experiment.</p>
           <span class="go">Prepare your data <span aria-hidden="true">→</span></span>
         </a></li>
         <li><a class="kf-nextcard kf-h-steel" href="usage/registration.html" aria-labelledby="task-registration">
           <span class="intent">Image registration</span>
           <h3 id="task-registration">Align two images</h3>
           <p>Choose an IMPACT-Reg preset and register a moving image to a fixed image.</p>
           <span class="go">Use IMPACT-Reg <span aria-hidden="true">→</span></span>
         </a></li>
         <li><a class="kf-nextcard kf-h-violet" href="usage/studio.html" aria-labelledby="task-studio">
           <span class="intent">Interactive workflows</span>
           <h3 id="task-studio">Work through a conversation</h3>
           <p>Inspect data, run jobs and view their results with Studio and the MCP server.</p>
           <span class="go">Open the Studio guide <span aria-hidden="true">→</span></span>
         </a></li>
       </ul>
       <p class="kf-start-client">Already using an MCP client? <a href="usage/mcp.html">Connect the MCP server</a>.</p>
     </section>
     <section class="kf-docindex">
       <div class="dhead"><h2>Keep these nearby</h2><span>Settings, examples and answers</span></div>
       <div class="kf-doclinks">
         <a href="config_guide/index.html">YAML configuration</a>
         <a href="reference/cli.html">CLI reference</a>
         <a href="usage/python-api.html">Python API</a>
         <a href="examples/index.html">Example catalogue</a>
         <a href="troubleshooting.html">Troubleshooting</a>
         <a href="reference/glossary.html">Glossary</a>
       </div>
     </section>
   </div>

.. toctree::
   :maxdepth: 1
   :caption: Getting started
   :hidden:

   getting-started/installation
   Run your first app <usage/apps>
   Train, predict and evaluate <quickstart>
   troubleshooting

.. toctree::
   :maxdepth: 1
   :caption: Task guides
   :hidden:

   usage/making-data
   usage/adopting-konfai
   usage/registration
   usage/large-images
   usage/studio
   usage/mcp
   usage/packaging-apps

.. toctree::
   :maxdepth: 1
   :caption: Examples
   :hidden:

   examples/index
   examples/visual-gallery
   examples/transform
   examples/segmentation
   Train a registration model <examples/registration>
   examples/synthesis

.. toctree::
   :maxdepth: 1
   :caption: Reference
   :hidden:

   config_guide/index
   config_guide/training
   config_guide/prediction
   config_guide/evaluation
   config_guide/transform
   reference/cli
   usage/python-api
   reference/components/models
   reference/components/losses-metrics
   reference/components/transforms
   reference/components/storage-backends
   reference/app-server-api
   reference/glossary

.. toctree::
   :maxdepth: 1
   :caption: Internals and contributing
   :hidden:

   usage/custom-models
   reference/api/index
   development
