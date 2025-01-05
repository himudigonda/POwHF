from abc import ABC, abstractmethod
from powhf import llm, utils


def get_eval_method(eval_method):
    """Returns the evaluation method object."""
    utils.debug_log(
        f"powhf.evaluate.get_eval_method :: Getting eval method: {eval_method}"
    )
    if callable(eval_method):
        utils.debug_log("powhf.evaluate.get_eval_method :: Eval method is callable")
        return eval_method
    elif eval_method == "likelihood":
        from powhf.evaluate import likelihood

        utils.debug_log("powhf.evaluate.get_eval_method :: Eval method is likelihood")
        return likelihood.likelihood_evaluator
    elif eval_method == "bandits":
        from powhf.evaluate import bandits

        utils.debug_log("powhf.evaluate.get_eval_method :: Eval method is bandits")
        return bandits.bandits_evaluator
    else:
        utils.debug_log(
            f"powhf.evaluate.get_eval_method :: Invalid eval method: {eval_method}"
        )
        raise ValueError(f"Invalid evaluation method: {eval_method}")


def evalute_prompts(
    prompts,
    eval_template,
    eval_data,
    demos_template,
    few_shot_data,
    eval_method,
    config,
):
    """Evaluates the prompts."""
    utils.debug_log(
        f"powhf.evaluate.evalute_prompts :: Evaluating {len(prompts)} prompts"
    )
    eval_method = get_eval_method(eval_method)
    res = eval_method(
        prompts, eval_template, eval_data, demos_template, few_shot_data, config
    )
    utils.debug_log("powhf.evaluate.evalute_prompts :: Finished evaluating prompts")
    return res


def demo_function(eval_template, config):
    """Returns a function for manual testing of prompts."""
    utils.debug_log("powhf.evaluate.demo_function :: Creating demo function")
    model = llm.model_from_config(config["model"])

    def fn(prompt, inputs):
        utils.debug_log(
            f"powhf.evaluate.demo_function.fn :: Testing prompt, num inputs: {len(inputs)}"
        )
        if not isinstance(inputs, list):
            inputs = [inputs]
        queries = []
        for input_ in inputs:
            query = eval_template.fill(prompt=prompt, input=input_)
            queries.append(query)
        outputs = model.generate_text(queries, n=1)
        outputs = [out.strip().split("\n")[0] for out in outputs]
        utils.debug_log(f"powhf.evaluate.demo_function.fn :: Outputs: {outputs}")
        return outputs

    return fn


class EvaluationResult(ABC):
    """Abstract base class for evaluation results."""

    @abstractmethod
    def sorted(self, method="default"):
        pass

    @abstractmethod
    def in_place(self, method="default"):
        pass
