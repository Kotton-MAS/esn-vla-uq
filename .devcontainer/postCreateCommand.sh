#!/usr/bin/env sh
echo 'source /usr/share/bash-completion/completions/git' >> ~/.bashrc
# Makefile の `sync` と同じくロックを固定する。固定しないと初回起動で
# uv.lock が無自覚に書き換わり、`make ci` 先頭の `uv lock --check` が落ちる (U4)。
uv sync --locked
