#!/bin/bash
# Script to trigger IR dump workflows on both branches and compare results
#
# Usage: ./scripts/run_ir_comparison.sh

set -e

MAIN_BRANCH="main"
PINUPDATE_BRANCH="quinnlp/pin-update"

echo "==================================="
echo "IR Comparison Workflow Runner"
echo "==================================="
echo ""

# Check if gh CLI is available
if ! command -v gh &> /dev/null; then
    echo "ERROR: GitHub CLI (gh) is not installed"
    echo "Install with: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    echo "ERROR: Not authenticated with GitHub CLI"
    echo "Run: gh auth login"
    exit 1
fi

echo "Step 1: Triggering workflow on $MAIN_BRANCH..."
MAIN_RUN_ID=$(gh workflow run debug-tdesc-ir.yml \
    --ref "$MAIN_BRANCH" \
    -f branch_name="main" \
    --json url,databaseId -q '.databaseId' 2>&1)

if [ $? -ne 0 ]; then
    echo "Failed to trigger main workflow"
    echo "$MAIN_RUN_ID"
    exit 1
fi

echo "  Triggered: https://github.com/intel/intel-xpu-backend-for-triton/actions/runs/$MAIN_RUN_ID"
echo ""

echo "Step 2: Triggering workflow on $PINUPDATE_BRANCH..."
PINUPDATE_RUN_ID=$(gh workflow run debug-tdesc-ir.yml \
    --ref "$PINUPDATE_BRANCH" \
    -f branch_name="pin-update" \
    --json url,databaseId -q '.databaseId' 2>&1)

if [ $? -ne 0 ]; then
    echo "Failed to trigger pin-update workflow"
    echo "$PINUPDATE_RUN_ID"
    exit 1
fi

echo "  Triggered: https://github.com/intel/intel-xpu-backend-for-triton/actions/runs/$PINUPDATE_RUN_ID"
echo ""

echo "==================================="
echo "Workflows triggered!"
echo ""
echo "Monitor progress with:"
echo "  gh run watch $MAIN_RUN_ID"
echo "  gh run watch $PINUPDATE_RUN_ID"
echo ""
echo "Once complete, download and compare with:"
echo "  gh run download $MAIN_RUN_ID -n ir-dump-main -D /tmp/ir_dump_main"
echo "  gh run download $PINUPDATE_RUN_ID -n ir-dump-pin-update -D /tmp/ir_dump_pin-update"
echo "  ./scripts/compare_ir_dumps.sh"
echo "==================================="

# Optionally wait for completion
read -p "Wait for workflows to complete? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Waiting for main workflow..."
    gh run watch "$MAIN_RUN_ID" --exit-status || echo "Main workflow failed or was cancelled"

    echo ""
    echo "Waiting for pin-update workflow..."
    gh run watch "$PINUPDATE_RUN_ID" --exit-status || echo "Pin-update workflow failed or was cancelled"

    echo ""
    echo "Downloading artifacts..."
    gh run download "$MAIN_RUN_ID" -n ir-dump-main -D /tmp/ir_dump_main
    gh run download "$PINUPDATE_RUN_ID" -n ir-dump-pin-update -D /tmp/ir_dump_pin-update

    echo ""
    echo "Running comparison..."
    ./scripts/compare_ir_dumps.sh
fi
