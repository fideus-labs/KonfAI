# Visual asset provenance

The figures listed here come from traceable scientific datasets and executed
workflows. They are not synthetic stock images.

## SynthRAD medical-image attribution and licence

- Source dataset: Adrian Thummerer et al., *SynthRAD2025 Grand Challenge
  dataset: Generating synthetic CTs for radiotherapy from head to abdomen*,
  [DOI 10.1002/mp.17981](https://doi.org/10.1002/mp.17981), distributed from
  [the official SynthRAD 2025 data page](https://synthrad2025.grand-challenge.org/data/).
- Source cases: `1ABB124` for synthesis/segmentation/evaluation/uncertainty and
  `1ABB123` for registration. Both are Task 1 abdomen, centre B training cases.
- Licence: [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).
- Changes: extraction of specified physical planes, display windowing and
  cropping, model-derived sCT/labels/maps, fixed-CT contours, and displacement
  vectors. Captions and metrics are HTML, not pixels embedded in the PNG files.

These medical-image derivatives retain the CC BY-NC 4.0 terms and are **not
covered by KonfAI's Apache-2.0 code licence**.

This attribution also covers `gallery/transforms/*.png` and
`gallery/augmentations/*.png`, which use the CT from case `1ABB124`.

## ExaSPIM microscopy attribution and licence

The OME-Zarr scale figure uses specimen `822175`, asset
`exaSPIM_822175_2026-03-27_16-38-42`, from the Allen Institute for Neural
Dynamics' MSMA Platform project (investigator: Jayaram Chandrashekar). Its
[AIND data description](https://aind-open-data.s3.amazonaws.com/exaSPIM_822175_2026-03-27_16-38-42/data_description.json)
declares CC BY 4.0 with no additional restriction. The displayed pyramid and
brain mask come from the corresponding public `aind-open-data` processed asset.
The two WebP derivatives retain [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
and are not covered by KonfAI's Apache-2.0 code licence.

## Source and run-artifact SHA-256

| Workflow | Artifact | SHA-256 |
| --- | --- | --- |
| ImpactSynth | `1ABB124/MR_IMPACT.mha` | `92f00f9aaa91265ba4408b635e55c1f645600af42ca5256c421a335a2787e7fe` |
| ImpactSynth | `1ABB124/CT.mha` | `bd7637303d4e8ac889ed3d59e1f3506301f72b64a17f4d7f798d9dd9ad3e0f30` |
| ImpactSynth | `sCT.mha` | `8557b497b8f00c994f0c4e5117c28a4aed66200dc0e2107fddc0d877c2b469d7` |
| TotalSegmentator | `Output.mha` (`total`, five models) | `b3cbe7fc5026fcbd9066ee2e0a9b8f21edc0ad75a408a375cdcafc9673f469ea` |
| ImpactSynth evaluation | `MAE_map.mha` | `dde2a03dc34b02c51da6ee9b1e910344f0dd3f6ec8e3be12a435d2b914e96385` |
| ImpactSynth evaluation | `Metric_TRAIN.json` | `93e4e1ed709579028f16a5a94d0ff6513b486f0d5ad370afa629844e14a40393` |
| ImpactSynth uncertainty | `Uncertainty.mha` | `1c8e8dbc28813fcd81682bd4b30aeb8d42dc4d6fd187c2949b49677f6c8fc653` |
| ImpactSynth uncertainty | `Metric_TRAIN.json` | `a2b37c8284daf01f08043725813a79d0b9f24a5c0840c32b653f1cef541a419f` |
| IMPACT-Reg | `1ABB123/MR_IMPACT.mha` | `7bf11f44b04d3b4c92302936d63f187b4282a6d8fd3bba0130ab015f7c21a135` |
| IMPACT-Reg | `1ABB123/CT.mha` | `aa716a2b7ae5bebba54a5a55d58af3dab0964330392c4725857161079869432e` |
| IMPACT-Reg | `MR_controlled-offset.mha` | `23063bd72ba0aade1f8f7cb32d38814098ff26efe44c4184f14d465f9ca2e946` |
| IMPACT-Reg | `Moved.mha` | `4ef0c1581c64ba64be1091706c6c5cf69ccc1c548346c393118ae1b39f93f1e1` |
| IMPACT-Reg | `DVF.mha` | `c3d1b8547bd6141d8e63fd78c533a98118504fbd7928a745366d7642518b7c25` |
| IMPACT-Reg | `Transform.h5` | `bc020e5ad58cc8234a333dc4f04cfb9faf73afd0327417287c0ab138498e30a7` |

The App evidence was produced with KonfAI `5195b79`, ImpactSynth `db04a8e`,
and ImpactReg `1e7ef81`. `generate_app_proof_gallery.py` uses axial index `84`
for MR/sCT/CT/labels/MAE, axial index `56` for uncertainty, and physical plane
`z=+18 mm`. `generate_registration_proof_gallery.py` uses coronal index `125`
and displays the coronal anatomy with superior at the top.

## Committed panel SHA-256

| Panel | SHA-256 |
| --- | --- |
| `impact-synth/mr-input.png` | `1e0bccfc2db3603f979a0dbc2847ebae6d2284e58d72ab52a76c6015e264e278` |
| `impact-synth/synthetic-ct.png` | `916b2a9f49c9fe669cc41c8bf74fa7eeba48dbc03697a9ea6513c15fac4518fe` |
| `impact-synth/reference-ct.png` | `b2ba3bf5294121ef3bc816d18e6bd364560e41e0dff07729d7aa130e52f18246` |
| `impact-synth/totalsegmentator.png` | `6ce932c1caa74a68e626b05506b680baba1c934fababd2d9fa7debee3a4b4d90` |
| `impact-synth/mae-map.png` | `a5bb0b5dbc012b7b3d635ee110afd3b32075c4fa50314a8e45b8812c3d92dbfd` |
| `impact-synth/uncertainty-map.png` | `68571f0f2f7cf74a2351b62d297024bd8830b44acc00d3f48b8b3e92a697e86c` |
| `impact-reg/moving-before.png` | `22e6b3505e19ba6bbf7ca0ff9aace5be355cb344c4b92bcbe9c034a1a0008fa9` |
| `impact-reg/fixed-ct.png` | `145b493d91915836f67f4750664f662cd300d2b3126e212f616e2ed8199cadbe` |
| `impact-reg/moved-after.png` | `4d0441d44520085cdd157625eae8a73b24877a358e77d95aed84ef866225643b` |
| `impact-reg/displacement-field.png` | `f19cab3fb2eb17c0e52c5a924b45ac1e6e324af3b58f4f499793d94c4952c6ea` |
| `gallery/scale-omezarr.webp` | `2e0e44c6e443af4117373aac971756b341ec6bd0a0af5ea159858b64f6f3e579` |
| `gallery/scale-omezarr-mobile.webp` | `02b602195ad5f42e6b8184816b66e678c0fe881bc580b32c81855cce26e9713b` |

The transform and augmentation panels come from axial index `84` of case
`1ABB124` (the `CT.mha` above). `docs/scripts/generate_visual_gallery.py
--input CT.mha --slice-index 84`, run with KonfAI `4d452cf` (v1.6.0), writes
these 17 files byte for byte; the stages have changed since, so another version
can draw them differently.

| Panel | SHA-256 |
| --- | --- |
| `gallery/transforms/source.png` | `d435a7cb2e29070a70b48ac0780c8366b3e078c1999d2d10b20be08677078fa7` |
| `gallery/transforms/clip.png` | `c35559d53d50cff3004e7ccc0648b26a8d77e6a9947d8f389d870be654b1f921` |
| `gallery/transforms/normalize.png` | `b9730a73f378584bf7e43e3f898a42c91b36dee59c158cdaa7e31d4dbfffe2d9` |
| `gallery/transforms/standardize.png` | `0ada4e44596316e673a1ab5be89ecbaf5258bd3126971bec9d1ea8e4850fb853` |
| `gallery/transforms/resample-shape.png` | `7427df0122b9a7eaec0807e010f195168ff9dbbc434cc64b0c8bc4b6a3d02f35` |
| `gallery/transforms/resample-spacing.png` | `1b59aca2ae926efcdf8de354b5c03599f34d87e7e5149477bbdd11bcd5b6ef86` |
| `gallery/transforms/padding.png` | `7ce0f833ba91cb2e93253afac7f74cc3091bcfda66b6d1aa813914a14990ebb2` |
| `gallery/transforms/crop.png` | `81e2cb0ac8400d4eeaa546633c1548692757e803d4a4e1307c505eb769785c0d` |
| `gallery/transforms/permute.png` | `8596ad620e2081da4cd4e8f975a769f719e948306d7b36af059ac84563c5c61a` |
| `gallery/transforms/gradient.png` | `ca8039d1420e7bfc236a3e4b67231fd40bbdf9dc4eb3e7885da6920606b9ebf4` |
| `gallery/augmentations/source.png` | `b9730a73f378584bf7e43e3f898a42c91b36dee59c158cdaa7e31d4dbfffe2d9` |
| `gallery/augmentations/flip.png` | `73b67999f4ef87258abf61ed4ec93aed4e6a316bf5dd5f01e0a1d50e547c9d71` |
| `gallery/augmentations/rotate.png` | `f932fcaf5efdd08f6902c0a1d81ed95238ba60ab33c4b2e857e4272f4ae6ba93` |
| `gallery/augmentations/brightness.png` | `99c4832055269a7a28ab01928a2aa43e36aa2d18eb81ceed1c88155d93e6986d` |
| `gallery/augmentations/contrast.png` | `29efc71f5b814e1017b622245a9fa6cf672eddd4060220e6cdfa1d4b113bae9c` |
| `gallery/augmentations/noise.png` | `994cbe14b83e46ddf947d3525c9220ab3ff1768eb49152df5c2e0ff53f0fec26` |
| `gallery/augmentations/cutout.png` | `62c7ec9333a1ca79de392e4904c507d02b89a17da0e989094b56981e293a6d49` |

The generators request plane-sized images through SimpleITK's extraction API.
Because the source MHA files are compressed, the backend may still decompress
their payload internally; these figures are not evidence of regional storage
I/O performance.
