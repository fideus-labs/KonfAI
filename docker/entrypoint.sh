#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
set -eu

if [ "$#" -eq 0 ]; then
    set -- konfai --help
fi

# A program on the PATH (konfai, konfai-apps, python, bash) runs as given; anything else, a
# subcommand such as TRAIN or an option, is an argument of konfai.
case "$(command -v -- "$1" || true)" in
    /*)
        exec "$@"
        ;;
    *)
        exec konfai "$@"
        ;;
esac
