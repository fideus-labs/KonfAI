# Copyright (c) 2025 Valentin Boussot
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""The shipped ViT.yml Vision Transformer catalog entry.

Validation level: structural-strict PLUS verified numerical equivalence of the encoder to MONAI 1.4.0
``ViT(classification=False)``. The structural asserts (build, forward on 2D and 3D, output shapes, terminal
head, parameter band) run WITHOUT MONAI so CI validates the entry; the MONAI oracle path is guarded by
``pytest.importorskip`` and proves the transformer maths match token-for-token.

Numerical equivalence is demonstrated by transferring MONAI's weights into the KonfAI graph in
forward-execution order (the mechanism of ``konfai.utils.pretrained``) and copying the single positional
embedding, after which the normalised token features match ``torch.allclose``.

Documented divergences from MONAI's ViT (why the shipped ``transfer_weights_by_execution_order`` cannot be
called on the raw pair, and why the classifier is not token-identical):
  * the learnable positional embedding is a standalone ``PositionalEmbedding`` leaf here, whereas MONAI
    stores it as a parameter on the non-leaf ``PatchEmbeddingBlock``; the leaf-pairing bridge therefore
    counts one extra leaf on the KonfAI side and refuses the pair, so the equivalence test transfers the
    shared leaves and copies the positional embedding explicitly;
  * classification uses global-average-pooling over the token sequence instead of a prepended ``cls`` token
    (MONAI's ``classification=True`` default) -- the ENCODER is identical, only the head differs;
  * MONAI 1.4.0's transformer block allocates unused cross-attention parameters that this graph omits.
"""

from pathlib import Path

import pytest
import torch
from konfai.network.blocks import MultiHeadSelfAttention, PositionalEmbedding
from konfai.network.network import Network
from konfai.utils.errors import ConfigError
from konfai.utils.model_builder import build_model_from_yaml, list_registered_modules
from konfai.utils.pretrained import (
    _parametric_leaves_in_execution_order,
    transfer_weights_by_execution_order,
)

CATALOG = Path(__file__).resolve().parents[2] / "konfai" / "models" / "yaml"
VIT_YML = CATALOG / "ViT.yml"

HIDDEN = 64
NUM_HEADS = 8
MLP_DIM = 128
NUM_LAYERS = 4
NUM_CLASSES = 2
IMG = 32
PATCH = 16


def _build_vit(dim: int, num_tokens: int) -> Network:
    return build_model_from_yaml(yaml_path=str(VIT_YML), parameters={"dim": dim, "num_tokens": num_tokens})


def _encoder_features(net: Network, inputs: torch.Tensor) -> torch.Tensor:
    features = None
    with torch.no_grad():
        for name, out in net.named_forward(inputs):
            if name == "Encoder.Norm":
                features = out
    assert features is not None, "the ViT graph never produced an 'Encoder.Norm' token-feature output"
    return features


def test_vit_builds_as_network() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    assert isinstance(net, Network)
    assert net.name == "ViT"


def test_vit_registry_primitives_are_registered() -> None:
    registered = set(list_registered_modules())
    assert {"PositionalEmbedding", "MultiHeadSelfAttention"} <= registered


@pytest.mark.parametrize(
    "dim,input_shape,num_tokens",
    [(2, (2, 1, IMG, IMG), 4), (3, (2, 1, IMG, IMG, IMG), 8)],
    ids=["2d", "3d"],
)
def test_vit_forward_shapes(dim: int, input_shape: tuple[int, ...], num_tokens: int) -> None:
    net = _build_vit(dim=dim, num_tokens=num_tokens)
    net.eval()
    inputs = torch.randn(*input_shape)

    logits = net.forward_tensor(inputs)
    # KonfAI classifier convention: [B, num_classes, 1].
    assert logits.shape == (input_shape[0], NUM_CLASSES, 1)

    features = _encoder_features(net, inputs)
    # Normalised token sequence: [B, num_tokens, hidden_size].
    assert features.shape == (input_shape[0], num_tokens, HIDDEN)


def test_vit_terminal_head_is_marked() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    terminal = [name for name, _, args in net.named_module_args_dict() if "-1" in args.out_branch]
    assert terminal == ["Head"]


def test_vit_has_four_encoder_layers_each_with_self_attention() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    attentions = [m for m in net.modules() if isinstance(m, MultiHeadSelfAttention)]
    positional = [m for m in net.modules() if isinstance(m, PositionalEmbedding)]
    assert len(attentions) == NUM_LAYERS
    assert len(positional) == 1


def test_vit_parameter_count_in_sane_band() -> None:
    net = _build_vit(dim=3, num_tokens=8)
    n_params = sum(p.numel() for p in net.parameters())
    # 4 pre-norm layers at hidden=64, mlp=128 land around ~4e5 parameters.
    assert 100_000 < n_params < 5_000_000


def test_vit_encoder_is_numerically_equivalent_to_monai() -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import ViT as MonaiViT

    net = _build_vit(dim=3, num_tokens=8)
    net.eval()
    reference = MonaiViT(
        in_channels=1,
        img_size=(IMG, IMG, IMG),
        patch_size=(PATCH, PATCH, PATCH),
        hidden_size=HIDDEN,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        classification=False,
        spatial_dims=3,
    ).eval()

    torch.manual_seed(0)
    inputs = torch.randn(1, 1, IMG, IMG, IMG)

    def encoder_forward() -> None:
        for name, _ in net.named_forward(inputs):
            if name == "Encoder.Norm":
                break

    # Pair the weighted leaves by forward-execution order, exactly as the pretrained bridge does. The
    # positional embedding is skipped on the KonfAI side because MONAI keeps it as a non-leaf parameter.
    target_leaves = [
        module
        for module in _parametric_leaves_in_execution_order(net, encoder_forward)
        if not isinstance(module, PositionalEmbedding)
    ]
    source_leaves = _parametric_leaves_in_execution_order(reference, lambda: reference(inputs))
    assert len(target_leaves) == len(source_leaves) > 0

    for target_leaf, source_leaf in zip(target_leaves, source_leaves, strict=True):
        target_shapes = {key: tuple(value.shape) for key, value in target_leaf.state_dict().items()}
        source_shapes = {key: tuple(value.shape) for key, value in source_leaf.state_dict().items()}
        assert target_shapes == source_shapes
        target_leaf.load_state_dict(source_leaf.state_dict())

    positional = next(module for module in net.modules() if isinstance(module, PositionalEmbedding))
    with torch.no_grad():
        positional.positional_embedding.copy_(reference.patch_embedding.position_embeddings)

    features = _encoder_features(net, inputs)
    reference_features, _ = reference(inputs)
    assert features.shape == reference_features.shape == (1, 8, HIDDEN)
    assert torch.allclose(features, reference_features, atol=1e-5), (features - reference_features).abs().max().item()


def test_shipped_bridge_refuses_the_raw_pair_because_of_the_positional_embedding() -> None:
    pytest.importorskip("monai")
    from monai.networks.nets import ViT as MonaiViT

    net = _build_vit(dim=3, num_tokens=8)
    net.eval()
    reference = MonaiViT(
        in_channels=1,
        img_size=(IMG, IMG, IMG),
        patch_size=(PATCH, PATCH, PATCH),
        hidden_size=HIDDEN,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        classification=False,
        spatial_dims=3,
    ).eval()
    inputs = torch.randn(1, 1, IMG, IMG, IMG)

    # The KonfAI graph carries the positional-embedding (and classifier-head) weights as extra leaves the
    # MONAI feature encoder does not expose, so the execution-order bridge honestly refuses the pair.
    with pytest.raises(ConfigError, match="different number of weighted leaves"):
        transfer_weights_by_execution_order(
            target=net,
            source=reference,
            target_forward=lambda: list(net.named_forward(inputs)),
            source_forward=lambda: reference(inputs),
        )
