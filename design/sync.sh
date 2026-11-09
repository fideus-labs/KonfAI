#!/usr/bin/env sh
# SPDX-License-Identifier: Apache-2.0
# Copies the identity kit to the places that serve or build it. design/ is the source; run this after editing it
# and commit the copies with the change. The site takes everything; Studio takes the fonts and the tokens.
set -eu
cd "$(dirname "$0")/.."
rsync -a --delete --exclude README.md --exclude sync.sh design/ site/design/
rsync -a --delete --include 'fonts/***' --include fonts.css --include tokens.css --exclude '*' design/ konfai-studio/frontend/src/design/
echo "design/ -> site/design/ and konfai-studio/frontend/src/design/"
