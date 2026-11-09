#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: $(basename "${BASH_SOURCE[0]}") <dir> <arg1> [arg2 ...]" >&2
  exit 1
fi

dir=$1
shift

patterns=()
for arg in "$@"; do
  patterns+=(-e "\"$arg\"")
done

grep -rlF "${patterns[@]}" -- "$dir"
