# Bundled calibrators

`TGPProbe.from_pretrained(model, calibrator="auto")` looks here for a file named
after the model: `/` becomes `--` and the name is lowercased.

    Qwen/Qwen2.5-3B-Instruct  ->  qwen--qwen2.5-3b-instruct.json

To add one, run this on a GPU and commit the resulting JSON:

    tgp-probe train --model Qwen/Qwen2.5-3B-Instruct \
        --out src/tgp_probe/calibrators/qwen--qwen2.5-3b-instruct.json
