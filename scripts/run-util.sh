print_env_info() {
    conda info
    conda list -n triton
}
set -vx
while [ -v 1 ]; do
  case "$1" in
    --python-version)
      python_version=$2
      shift 2
      ;;
    *)
      script_name=$1
      shift
      ;;
  esac
done

if [ $(head -1 "$script_name") = install-env ]
then
  install_env
  print_env_info
  exit 0
fi
