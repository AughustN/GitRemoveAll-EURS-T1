The weights are not committed. `weights_link.yaml` points to one public zip on Hugging Face
(`eurs_t1_weights.zip`, 1.57 GB). Fetch and unpack it here with

    python repro_check.py . --download-weights

`run.py` gets this folder as `model_dir` and finds the files below it by name.
