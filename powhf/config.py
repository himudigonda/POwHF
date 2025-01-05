import os
import yaml
from powhf import utils


def update_config(config, base_config="powhf/configs/default.yaml"):
    """Updates the default config with user-provided config."""
    utils.debug_log(f"powhf.config.update_config :: Loading base config: {base_config}")
    with open(os.path.join(os.path.dirname(__file__), base_config)) as f:
        default_config = yaml.safe_load(f)

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    updated_config = update(default_config, config)
    utils.debug_log(f"powhf.config.update_config :: Updated config: {updated_config}")
    return updated_config


def simple_config(
    eval_model,
    prompt_gen_model,
    prompt_gen_mode,
    num_prompts,
    eval_rounds,
    prompt_gen_batch_size,
    eval_batch_size,
):
    """Creates a simple configuration."""
    utils.debug_log("powhf.config.simple_config :: Creating simple config")
    conf = update_config({}, "powhf/configs/bandits.yaml")
    conf["generation"]["model"]["model_name"] = prompt_gen_model
    if prompt_gen_mode == "insert":
        conf["generation"]["model"]["name"] = "HF_insert"
        conf["generation"]["model"]["batch_size"] = 1
    elif prompt_gen_mode == "forward":
        conf["generation"]["model"]["name"] = "HF_forward"
        conf["generation"]["model"]["batch_size"] = prompt_gen_batch_size
    conf["generation"]["num_subsamples"] = num_prompts // 10
    conf["generation"]["num_prompts_per_subsample"] = 10

    conf["evaluation"]["base_eval_config"]["model"]["model_name"] = eval_model
    conf["evaluation"]["base_eval_config"]["model"]["batch_size"] = eval_batch_size
    conf["evaluation"]["num_prompts_per_round"] = 0.334
    conf["evaluation"]["rounds"] = eval_rounds
    conf["evaluation"]["base_eval_config"]["num_samples"] = 5
    utils.debug_log(f"powhf.config.simple_config :: Simple config: {conf}")
    return conf
