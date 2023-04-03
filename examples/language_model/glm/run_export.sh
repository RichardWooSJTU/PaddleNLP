  export PYTHONPATH=${PWD}:${PYTHONPATH}
  FUSE_MT=1 python3 examples/language_model/glm/export_generation_model.py --model_path /root/paddlejob/workspace/env_run/fhq/models/glm_10b