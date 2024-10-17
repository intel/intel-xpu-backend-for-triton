#/bin/sh

test -d pytorch || {
  git clone https://github.com/pytorch/pytorch
  cd pytorch
  TRANSFORMERS_VERSION=$(.ci/docker/ci_commit_pins/huggingface.txt)
  pip install transformers=$TRANSFORMERS_VERSION
  python -c "import transformers; print(transformers.__version__)"
}

pip install pyyaml pandas scipy numpy psutil pyre_extensions torchrec

ZE_AFFINITY_MASK=0 python pytorch/benchmarks/dynamo/huggingface.py --accuracy --float32 -dxpu -n10 --no-skip --dashboard --inference --freezing --total-partitions 1 --partition-id 0 --only AlbertForMaskedLM --backend=inductor --timeout=4800 --output=inductor_log.csv

cat inductor_log.csv
grep AlbertForMaskedLM inductor_log.csv | grep -q ,pass,
