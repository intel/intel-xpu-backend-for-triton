#!/usr/bin/env bash
# auto-upstream-merge.sh — Attempt to merge new upstream triton-lang/triton
# commits into the Intel XPU backend fork.
#
# This is a standalone helper for LOCAL TESTING ONLY. It is NOT invoked by the
# auto-upstream-merge.yml GitHub Actions workflow, which implements the full
# pipeline (merge → build → pre-commit → tests → tracking file update → PR)
# inline. This script only performs the merge step and reports the result.
#
# NOTE: This script does NOT update upstream-triton-hash.txt. Per the workflow
# design, the tracking file is only updated after build, pre-commit, and unit
# tests all pass — checks that this script does not perform.
#
# Usage:
#   scripts/auto-upstream-merge.sh [--target-hash <hash>]
#
# Environment variables (set manually or by CI):
#   GITHUB_ENV — If set, outputs are appended here for GitHub Actions.
#                If unset, outputs are printed to stdout.
#
# Outputs (via $GITHUB_ENV or stdout):
#   CURRENT_HASH     — The upstream hash we're currently based on.
#   TARGET_HASH      — The upstream hash we're merging to.
#   TARGET_SHORT     — Short (7-char) version of TARGET_HASH.
#   MERGE_STATUS     — One of: "up_to_date", "clean", "conflicts", "error".
#   CONFLICT_FILES   — Newline-separated list of conflicting files (if any).
#   CONFLICT_COUNT   — Number of conflicting files.
#   PR_TITLE         — Suggested PR title matching existing convention.
#   PR_BODY          — Suggested PR body matching existing convention.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRACKING_FILE="${REPO_ROOT}/upstream-triton-hash.txt"
UPSTREAM_REPO="https://github.com/triton-lang/triton.git"
UPSTREAM_REMOTE="upstream-triton"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

emit() {
    local key="$1" value="$2"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        echo "${key}=${value}" >> "$GITHUB_ENV"
    fi
    echo "${key}=${value}"
}

# For multi-line values in GITHUB_ENV
emit_multiline() {
    local key="$1" value="$2"
    if [[ -n "${GITHUB_ENV:-}" ]]; then
        local delimiter
        delimiter="EOF_$(date +%s%N)"
        {
            echo "${key}<<${delimiter}"
            echo "${value}"
            echo "${delimiter}"
        } >> "$GITHUB_ENV"
    fi
    echo "${key}=$(echo "${value}" | head -5)..."
}

die() {
    echo "ERROR: $*" >&2
    emit "MERGE_STATUS" "error"
    exit 1
}

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------

TARGET_OVERRIDE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target-hash)
            if [[ $# -lt 2 || -z "$2" ]]; then
                die "Missing value for --target-hash; expected a commit hash."
            fi
            TARGET_OVERRIDE="$2"
            shift 2
            ;;
        *)
            die "Unknown argument: $1"
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Read current upstream base
# ---------------------------------------------------------------------------

if [[ ! -f "${TRACKING_FILE}" ]]; then
    die "Tracking file not found: ${TRACKING_FILE}"
fi

CURRENT_HASH="$(tr -d '[:space:]' < "${TRACKING_FILE}")"
if [[ -z "${CURRENT_HASH}" ]]; then
    die "Tracking file is empty: ${TRACKING_FILE}"
fi
emit "CURRENT_HASH" "${CURRENT_HASH}"

# ---------------------------------------------------------------------------
# Fetch upstream
# ---------------------------------------------------------------------------

echo "==> Fetching upstream from ${UPSTREAM_REPO}..."
if ! git remote get-url "${UPSTREAM_REMOTE}" &>/dev/null; then
    git remote add "${UPSTREAM_REMOTE}" "${UPSTREAM_REPO}"
fi
git fetch "${UPSTREAM_REMOTE}" main --quiet

# ---------------------------------------------------------------------------
# Determine target hash
# ---------------------------------------------------------------------------

if [[ -n "${TARGET_OVERRIDE}" ]]; then
    TARGET_HASH="${TARGET_OVERRIDE}"
else
    TARGET_HASH="$(git rev-parse "${UPSTREAM_REMOTE}/main")"
fi
TARGET_SHORT="${TARGET_HASH:0:7}"

emit "TARGET_HASH" "${TARGET_HASH}"
emit "TARGET_SHORT" "${TARGET_SHORT}"

# ---------------------------------------------------------------------------
# Check if there are new commits
# ---------------------------------------------------------------------------

if [[ "${CURRENT_HASH}" == "${TARGET_HASH}" ]]; then
    echo "==> Already up to date with upstream (${TARGET_SHORT})."
    emit "MERGE_STATUS" "up_to_date"
    exit 0
fi

NEW_COMMIT_COUNT="$(git rev-list "${CURRENT_HASH}..${TARGET_HASH}" --count 2>/dev/null || echo "?")"
echo "==> ${NEW_COMMIT_COUNT} new upstream commit(s): ${CURRENT_HASH:0:7}..${TARGET_SHORT}"

# ---------------------------------------------------------------------------
# Attempt the merge
# ---------------------------------------------------------------------------

echo "==> Attempting merge of ${TARGET_SHORT}..."

# Enable rerere for automatic conflict resolution reuse
git config rerere.enabled true
git config rerere.autoupdate true

if git merge "${TARGET_HASH}" --no-edit 2>/dev/null; then
    echo "==> Merge succeeded cleanly."
    emit "MERGE_STATUS" "clean"
    emit "CONFLICT_FILES" ""
    emit "CONFLICT_COUNT" "0"
else
    echo "==> Merge produced conflicts."
    # Collect conflicting files
    CONFLICT_FILES="$(git diff --name-only --diff-filter=U 2>/dev/null || true)"
    CONFLICT_COUNT="$(echo "${CONFLICT_FILES}" | grep -c '.' || echo 0)"

    emit "MERGE_STATUS" "conflicts"
    emit_multiline "CONFLICT_FILES" "${CONFLICT_FILES}"
    emit "CONFLICT_COUNT" "${CONFLICT_COUNT}"

    echo "==> ${CONFLICT_COUNT} file(s) with conflicts:"
    echo "${CONFLICT_FILES}" | sed 's/^/  - /'
fi

# ---------------------------------------------------------------------------
# Build PR metadata
# ---------------------------------------------------------------------------

MERGE_DATE="$(date +'%b %-d')"
PR_TITLE="Merge OpenAI Triton commit \`${TARGET_SHORT}\`"
PR_BODY="This PR changes the Triton base from
${CURRENT_HASH} to
${TARGET_HASH} (${MERGE_DATE}).

*Pass rate will be determined by CI.*

---
*Automated by \`auto-upstream-merge\` workflow.*"

emit "PR_TITLE" "${PR_TITLE}"
emit_multiline "PR_BODY" "${PR_BODY}"

echo "==> Done. MERGE_STATUS=$(git diff --name-only --diff-filter=U 2>/dev/null | grep -q . && echo conflicts || echo clean)"
