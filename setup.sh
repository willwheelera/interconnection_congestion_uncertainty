#!/usr/bin/env bash
set -e

if [ ! -d "Egret" ]; then
    echo "Cloning EGRET..."
    git clone https://github.com/grid-parity-exchange/Egret.git
else
    echo "Egret/ already exists, skipping clone."
fi

uv sync
