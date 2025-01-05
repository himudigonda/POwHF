from powhf import data, llm, utils


def get_query(prompt_gen_template, demos_template, subsampled_data):
    """Returns a query for the prompt generator."""
    utils.debug_log("powhf.generate.get_query :: Creating prompt generation query")
    inputs, outputs = subsampled_data
    demos = demos_template.fill(subsampled_data)
    query = prompt_gen_template.fill(
        input=inputs[0], output=outputs[0], full_demo=demos
    )
    utils.debug_log(f"powhf.generate.get_query :: Query: {query}")
    return query


def generate_prompts(prompt_gen_template, demos_template, prompt_gen_data, config):
    """Generates prompts using the provided LLM."""
    utils.debug_log(
        f"powhf.generate.generate_prompts :: Starting prompt generation, num subsamples: {config['num_subsamples']}"
    )
    queries = []
    for _ in range(config["num_subsamples"]):
        subsampled_data = data.subsample_data(prompt_gen_data, config["num_demos"])
        queries.append(get_query(prompt_gen_template, demos_template, subsampled_data))
    utils.debug_log(
        f"powhf.generate.generate_prompts :: Generated {len(queries)} queries"
    )
    model = llm.model_from_config(config["model"])
    prompts = model.generate_text(queries, n=config["num_prompts_per_subsample"])
    utils.debug_log(
        f"powhf.generate.generate_prompts :: Generated {len(prompts)} prompts"
    )
    return prompts
