# Common helper providing a `pip` wrapper that works both in traditional Python
# environments and in uv-managed ones. Environments created by `uv venv` do not
# contain pip, so packages have to be installed with `uv pip` there.
#
# Usage: source this file and call `pip` as usual, for example:
#
#   source "$(dirname "$0")/pip-utils.sh"
#   pip install ninja
#
# `pip` is resolved in the following order:
#   1. `python -m pip`, if pip is available in the active interpreter (this
#      keeps the behavior of traditional environments unchanged);
#   2. `uv pip`, if `uv` is in PATH (uv-managed environments without pip);
# if neither is available, the wrapper fails with an explanatory message.

# Resolved pip front-end, e.g. (python -m pip) or (uv pip). Empty until the
# first `pip` call.
PIP_CMD=()
# Virtual environment `PIP_CMD` was resolved for, to re-resolve the front-end if
# a virtual environment is activated (or deactivated) after this file is sourced.
# The initial value is not a valid path, so the first `pip` call always resolves.
PIP_CMD_VIRTUAL_ENV="<unresolved>"

# `python` is preferred over `python3`, because the scripts run `python`
# themselves, so packages have to be installed for that interpreter, and because
# `python3` does not necessarily exist on Windows. `python3` is used when
# `python` is missing or is Python 2.
pip_python_cmd() {
  if [ "$(python -c 'import sys; print(sys.version_info[0])' 2>/dev/null)" = 3 ]; then
    echo python
  else
    echo python3
  fi
}

pip_resolve_cmd() {
  local python_cmd
  python_cmd="$(pip_python_cmd)"

  if "$python_cmd" -m pip --version &>/dev/null; then
    PIP_CMD=("$python_cmd" -m pip)
  elif command -v uv &>/dev/null; then
    PIP_CMD=(uv pip)
  else
    echo "**** ERROR: neither pip nor uv is available, cannot install Python packages. ****" >&2
    echo "**** INFO: install pip ($python_cmd -m ensurepip) or uv (https://docs.astral.sh/uv/). ****" >&2
    return 1
  fi

  PIP_CMD_VIRTUAL_ENV="${VIRTUAL_ENV:-}"
  # Print to stderr, so that the message does not end up in the output of
  # commands like `$(pip show <package>)`.
  echo "**** Using '${PIP_CMD[*]}' to install Python packages ****" >&2
}

pip() {
  if [ "$PIP_CMD_VIRTUAL_ENV" != "${VIRTUAL_ENV:-}" ]; then
    pip_resolve_cmd || return 1
  fi
  "${PIP_CMD[@]}" "$@"
}
