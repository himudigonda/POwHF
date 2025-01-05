import random
from powhf import generate, evaluate, config, template, data, llm, utils
import logging


def simple_ape(
    dataset,
    eval_template="Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]",
    prompt_gen_template=None,
    demos_template="Input: [INPUT]\nOutput: [OUTPUT]",
    eval_model="google/flan-t5-base",  # set a default hf model
    prompt_gen_model="google/flan-t5-base",  # set a default hf model
    prompt_gen_mode="forward",
    num_prompts=50,
    eval_rounds=20,
    prompt_gen_batch_size=200,
    eval_batch_size=500,
):
    """
    Simplified function to wrap `find_prompts` for text-based tasks.
    """
    utils.debug_log(
        f"powhf.ape.simple_ape :: Starting APE process with dataset: {len(dataset[0])}"
    )

    prompt_gen_template = get_simple_prompt_gen_template(
        prompt_gen_template, prompt_gen_mode
    )
    conf = config.simple_config(
        eval_model,
        prompt_gen_model,
        prompt_gen_mode,
        num_prompts,
        eval_rounds,
        prompt_gen_batch_size,
        eval_batch_size,
    )

    res, demo_fn = find_prompts(
        eval_template,
        demos_template,
        dataset,
        dataset,
        conf,
        prompt_gen_template=prompt_gen_template,
    )
    utils.debug_log("powhf.ape.simple_ape :: Finished APE process")
    return res, demo_fn


def get_simple_prompt_gen_template(prompt_gen_template, prompt_gen_mode):
    """Helper function to get prompt template."""
    utils.debug_log(
        f"powhf.ape.get_simple_prompt_gen_template :: Getting simple prompt generation template, mode: {prompt_gen_mode}"
    )
    if prompt_gen_template is None:
        if prompt_gen_mode == "forward":
            prompt_gen_template = "I gave a friend an instruction. Based on the instruction they produced the following input-output pairs:\n\n[full_DEMO]\n\nThe instruction was to [APE]"
        elif prompt_gen_mode == "insert":
            prompt_gen_template = None
        else:
            raise ValueError(f"Invalid prompt_gen_mode: {prompt_gen_mode}")
    utils.debug_log(
        f"powhf.ape.get_simple_prompt_gen_template :: Template: {prompt_gen_template}"
    )
    return prompt_gen_template


def simple_eval(
    dataset,
    prompts,
    eval_template="Instruction: [PROMPT]\nInput: [INPUT]\nOutput: [OUTPUT]",
    demos_template="Input: [INPUT]\nOutput: [OUTPUT]",
    eval_model="google/flan-t5-base",
    num_samples=50,
):
    """Simplified evaluation of given prompts."""
    utils.debug_log(
        f"powhf.ape.simple_eval :: Starting simple eval, num prompts: {len(prompts)}, num samples: {num_samples}"
    )
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    conf = config.update_config({}, "powhf/configs/default.yaml")
    conf["evaluation"]["model"]["model_name"] = eval_model
    conf["evaluation"]["num_samples"] = min(len(dataset[0]), num_samples)
    res = evaluate.evalute_prompts(
        prompts,
        eval_template,
        dataset,
        demos_template,
        dataset,
        conf["evaluation"]["method"],
        conf["evaluation"],
    )
    utils.debug_log("powhf.ape.simple_eval :: Finished simple eval")
    return res


def find_prompts(
    eval_template,
    demos_template,
    prompt_gen_data,
    eval_data,
    conf,
    base_conf="powhf/configs/default.yaml",
    few_shot_data=None,
    prompt_gen_template=None,
):
    """
    Core function to generate and evaluate prompts.
    """
    utils.debug_log(
        "powhf.ape.find_prompts :: Starting prompt generation and evaluation"
    )
    conf = config.update_config(conf, base_conf)
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)

    if prompt_gen_template is None:
        prompt_gen_template = eval_template.convert_to_generation_template()
    else:
        prompt_gen_template = template.GenerationTemplate(prompt_gen_template)

    if few_shot_data is None:
        few_shot_data = prompt_gen_data

    prompts = generate.generate_prompts(
        prompt_gen_template, demos_template, prompt_gen_data, conf["generation"]
    )
    utils.debug_log(
        f"powhf.ape.find_prompts :: Generated {len(prompts)} prompts, deduplicating"
    )
    prompts = list(set(prompts))
    utils.debug_log(f"powhf.ape.find_prompts :: Deduplicated to {len(prompts)} prompts")

    res = evaluate.evalute_prompts(
        prompts,
        eval_template,
        eval_data,
        demos_template,
        few_shot_data,
        conf["evaluation"]["method"],
        conf["evaluation"],
    )

    utils.debug_log("powhf.ape.find_prompts :: Finished evaluating prompts")

    demo_fn = evaluate.demo_function(eval_template, conf["demo"])
    utils.debug_log("powhf.ape.find_prompts :: Finished find_prompts function")
    return res, demo_fn


def evaluate_prompts(
    prompts,
    eval_template,
    eval_data,
    demos_template,
    few_shot_data,
    conf,
    base_conf="powhf/configs/default.yaml",
):
    """Evaluates a list of prompts."""
    utils.debug_log(
        f"powhf.ape.evaluate_prompts :: Starting evaluation of {len(prompts)} prompts"
    )
    conf = config.update_config(conf, base_conf)
    eval_template = template.EvalTemplate(eval_template)
    demos_template = template.DemosTemplate(demos_template)
    res = evaluate.evalute_prompts(
        prompts,
        eval_template,
        eval_data,
        demos_template,
        few_shot_data,
        conf["evaluation"]["method"],
        conf["evaluation"],
    )
    utils.debug_log("powhf.ape.evaluate_prompts :: Finished evaluating prompts")
    return res
