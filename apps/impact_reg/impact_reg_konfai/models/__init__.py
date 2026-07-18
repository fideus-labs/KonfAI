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

"""IMPACT-Reg model engines, defined once and shared by every preset bundle.

Each submodule is a self-contained KonfAI registration model (a ``network.Network`` subclass named
``RegistrationNet`` plus its ``ModelSpec`` config schema): ``convexadam`` (itk-impact ConvexAdam),
``fireants`` (FireANTs), ``elastix`` (elastix parameter-map presets). A preset bundle carries only its
config + weights and points its ``classpath`` at one of these, e.g.
``impact_reg_konfai.models.convexadam:RegistrationNet``; a custom app imports the class and subclasses it.

This ``__init__`` deliberately imports NOTHING: each engine pulls a heavy, distinct backend (itk-impact,
fireants, elastix) at module import, so resolving one preset must not drag in the other two -- the
classpath imports only the single submodule it names.
"""
