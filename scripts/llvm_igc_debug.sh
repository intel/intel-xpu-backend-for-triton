#!/usr/bin/env bash

# Given a LLVM .ll or .bc file as input, this script dumps the corresponding IGC files.
# -o can be given to indicate desired output directory.
# -gen-spt can be given to generate SPIR-V text format output.
# -gen-ll can be given to generate LLVM IR output.
# Example usage:
# ./llvm_igc_debug.sh t.ll --gen-spt -o <output directory>

if [ ! -d "$BASE" ]; then
  echo "**** BASE is not given *****"
  echo "**** Default BASE is set to /iusers/$USER ****"
  BASE=/iusers/$USER
fi

LLVM_SPIRV=$BASE/packages//llvm-spirv/bin/llvm-spirv
if [ ! -f "$LLVM_SPIRV" ]; then
  echo "llvm-spirv $LLVM_SPIRV does not exist."
  exit 1
fi
LLVM_BIN=$BASE/packages/llvm/bin
if [ ! -d "$LLVM_BIN" ]; then
  echo "llvm bin directory $LLVM_BIN does not exist."
  exit 1
fi

OUT_DIR=$(pwd)
GEN_SPT=false
GEN_LL=false
while [[ $# -gt 0 ]]; do
  case $1 in
    -o)
      OUT_DIR="$2"
      if [ ! -d $OUT_DIR ]; then
        echo "Directory $OUT_DIR does not exist."
        exit 1
      fi
      shift
      shift
      ;;
    --gen-spt)
      GEN_SPT=true
      shift
      ;;
    --gen-ll)
      GEN_LL=true
      shift
      ;;
    --help)
      echo "Example usage: ./llvm_igc_debug.sh t.ll --gen-spt -o <output directory>"
      exit 1
      ;;
    *)
      if [ ! -z $FILE ]; then
        echo "File $FILE already given. Invalid option $1"
        exit 1
      fi
      FILE="${1}"
      shift
      ;;
  esac
done

if [ ! -f $FILE ]; then
  echo "File $FILE does not exist."
  exit 1
fi
FILENAME=$(basename $FILE)

FILE_BC=$(mktemp --suffix ".bc")
if [[ $FILENAME == *.bc ]]; then
  cp $FILE $FILE_BC
  if [ "$GEN_LL" = true]; then
    $LLVM_BIN/llvm-dis $FILE -o $OUT_DIR/"${FILENAME%.*}".ll
  fi
elif [[ $FILENAME == *.ll ]]; then
  $LLVM_BIN/llvm-as $FILE -o $FILE_BC
else
  echo "File $FILE invalid extension."
  exit 1
fi

# LLVM to SPIR-V translation
FILE_SPV=$(mktemp --suffix ".spv")
$LLVM_SPIRV $FILE_BC -o $FILE_SPV

# SPIR-V text format
if [ "$GEN_SPT" = true ]; then
  $LLVM_SPIRV --to-text $FILE_SPV -o $OUT_DIR/"${FILENAME%.*}".spt
fi

# ocloc compilation
export IGC_ShaderDumpEnableAll=1
export IGC_DumpToCustomDir=$OUT_DIR
ocloc compile -file $FILE_SPV -device pvc -spirv_input -options "-cl-opt-disable" -o $OUT_DIR/"${FILENAME%.*}".bin

rm $FILE_BC $FILE_SPV
