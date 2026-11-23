# Source this file to use the isolated server environment and its CUDA libraries.
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
export PATH="$TASK_ROOT/env/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TF_NUM_INTRAOP_THREADS="${TF_NUM_INTRAOP_THREADS:-8}"
export TF_NUM_INTEROP_THREADS="${TF_NUM_INTEROP_THREADS:-2}"
export PYTHONUNBUFFERED=1
CUDA_PIP_LIBS="$("$TASK_ROOT/env/bin/python" -c 'import glob,sys; print(":".join(glob.glob(sys.prefix+"/lib/python*/site-packages/nvidia/*/lib")))')"
export LD_LIBRARY_PATH="$CUDA_PIP_LIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
