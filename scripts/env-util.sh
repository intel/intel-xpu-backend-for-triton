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
