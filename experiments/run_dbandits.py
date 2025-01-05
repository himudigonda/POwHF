import os
import sys
import json
import argparse
from datetime import datetime
import torch
import numpy as np
import random
from powhf import ape, data, utils, template
from datasets import load_dataset
from powhf.evaluation.instruction_induction.utility import set_all_seed
from powhf.llm import model_from_config
from powhf.LlamaForMLPRegression import DoubleTS, LinearDBDiag, NeuralDBDiag
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F

cwd = os.getcwd()
sys.path.append(cwd)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
SMOKE_TEST = os.environ.get("SMOKE_TEST")

tkwargs = {
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    "dtype": torch.float32,
}


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def get_sen_embedding(model, tokenizer, sentences):
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        model_output = model(**encoded_input)
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    return sentence_embeddings


class LocalHFModelForwardAPI:
    def __init__(
        self,
        model_name,
        eval_data=None,
        init_prompt=None,
        init_qa_gen=None,
        conf=None,
        base_conf=None,
        prompt_gen_data=None,
        n_prompt_tokens=None,
        random_proj=None,
        intrinsic_dim=None,
        magnitude=None,
        norm_method=None,
    ):
        utils.debug_log(
            "experiments.run_dbandits.LocalHFModelForwardAPI :: Initializing LocalHFModelForwardAPI"
        )
        self.init_qa_gen = init_qa_gen
        self.init_prompt = init_prompt[0]
        init_qa = self.init_qa_gen()
        self.init_token = init_prompt[0] + init_qa
        self.count = 0

        self.conf = conf
        self.eval_data = eval_data
        self.eval_template = template.EvalTemplate(
            "Instruction: [PROMPT]\n\nInput: [INPUT]\n Output: [OUTPUT]"
        )
        self.demos_template = template.DemosTemplate("Input: [INPUT]\nOutput: [OUTPUT]")

        self.best_train_perf = 0.0
        self.best_dev_perf = 0.0
        self.best_last_perf = 10
        self.best_prompt = None
        self.num_call = 0
        self.best_instruction = None
        self.prompts_set = dict()
        self.prompts_list = []
        self.parents = []
        self.best_score = 0
        self.score_mean = None
        self.score_std = None
        self.score_min = None
        self.score_max = None
        self.magnitude = magnitude
        self.norm_method = norm_method
        self.init_user_prompt = None
        self.model_name = model_name

    def update_init_token(self):
        # randomly choose a qa
        init_qa = self.init_qa_gen()
        self.init_token = self.init_prompt + init_qa

    def initialize_prompts(self, num_init, task, method):
        utils.debug_log(
            f"experiments.run_dbandits.LocalHFModelForwardAPI.initialize_prompts :: Initializing prompts, num: {num_init}, method: {method}"
        )
        ini_prompts_his = {}
        model = model_from_config(self.conf["evaluation"]["model"])
        if method == "rephrase":
            model_outputs = model.generate_text(self.init_token, 1, 0.5)
            ini_prompts_his[model_outputs[0]] = 0
            self.init_user_prompt = model_outputs[0]
        while len(ini_prompts_his) < num_init:
            if method == "induction":
                if task in [
                    "sum",
                    "first_word_letter",
                    "periodic_elements",
                    "active_to_passive",
                ]:
                    random_prompt = model.generate_text(
                        self.init_token, 1, 1, use_seed=False
                    )[0]
                    model_outputs = model.generate_text(
                        "Rephrase the following instruction: "
                        + random_prompt
                        + "\n the rephrased instruction is: ",
                        1,
                        1,
                        use_seed=False,
                    )
                else:
                    model_outputs = model.generate_text(self.init_token, 1, 0.5)
                ini_prompts_his[model_outputs[0]] = 0
                self.update_init_token()
                utils.debug_log(
                    f"experiments.run_dbandits.LocalHFModelForwardAPI.initialize_prompts :: Task: {task}, num initialized: {len(ini_prompts_his)}"
                )
            elif method == "rephrase":
                if task in ["odd_one_out", "orthography_starts_with"]:
                    model_outputs = model.generate_text(
                        "Rephrase the following instruction: "
                        + self.init_user_prompt
                        + "\n the rephrased instruction is: ",
                        1,
                        1.5,
                        use_seed=False,
                    )
                else:
                    model_outputs = model.generate_text(
                        "Rephrase the following instruction: "
                        + self.init_user_prompt
                        + "\n the rephrased instruction is: ",
                        1,
                        1,
                        use_seed=False,
                    )
                ini_prompts_his[model_outputs[0]] = 0
                utils.debug_log(
                    f"experiments.run_dbandits.LocalHFModelForwardAPI.initialize_prompts :: Task: {task}, num initialized: {len(ini_prompts_his)}"
                )

        utils.debug_log(
            f"experiments.run_dbandits.LocalHFModelForwardAPI.initialize_prompts :: Returning initialized prompts, num: {len(ini_prompts_his)}"
        )
        return list(ini_prompts_his.keys())

    def eval(self, instruction, test=False):
        """Evaluates a given instruction."""
        utils.debug_log(
            f"experiments.run_dbandits.LocalHFModelForwardAPI.eval :: Evaluating instruction: {instruction}, test: {test}"
        )
        if instruction[0] in self.prompts_set.keys():
            dev_perf = self.prompts_set[instruction[0]]
        else:
            dev_perf, _ = ape.evaluate_prompts(
                instruction,
                self.eval_template,
                self.eval_data,
                self.demos_template,
                self.conf,
                self.conf,
            )
            dev_perf = dev_perf.sorted()[1][0]

            if not test:
                if dev_perf >= self.best_last_perf:
                    self.count += 1
                if dev_perf >= self.best_dev_perf:
                    self.best_dev_perf = dev_perf
                    self.best_instruction = instruction

                if self.norm_method == "standard":
                    dev_perf = (
                        self.magnitude * (dev_perf - self.score_mean) / self.score_std
                    )
                elif self.norm_method == "minmax":
                    dev_perf = (
                        self.magnitude
                        * (dev_perf - self.score_min)
                        / (self.score_max - self.score_min)
                    )
                self.prompts_set[instruction[0]] = dev_perf
                self.prompts_list.append(
                    (len(self.prompts_list), instruction[0], dev_perf)
                )
                utils.debug_log(
                    f"experiments.run_dbandits.LocalHFModelForwardAPI.eval :: Dev loss: {dev_perf}. Dev perf: {dev_perf}. Best dev perf: {self.best_dev_perf}"
                )
            utils.debug_log(
                f"experiments.run_dbandits.LocalHFModelForwardAPI.eval :: Returning dev_perf: {dev_perf}"
            )
        return dev_perf

    def return_best_prompt(self):
        """Returns the best prompt"""
        utils.debug_log(
            f"experiments.run_dbandits.LocalHFModelForwardAPI.return_best_prompt :: Returning best instruction: {self.best_instruction}"
        )
        return self.best_instruction

    def return_prompts_set(self):
        """Returns all prompt with scores"""
        utils.debug_log(
            f"experiments.run_dbandits.LocalHFModelForwardAPI.return_prompts_set :: Returning all prompts: {self.prompts_set}"
        )
        return self.prompts_set

    def return_prompts_list(self):
        """Returns list of prompts"""
        utils.debug_log(
            f"experiments.run_dbandits.LocalHFModelForwardAPI.return_prompts_list :: Returning list of prompts: {self.prompts_list}"
        )
        return self.prompts_list

    def get_sen_embedding(self, sen_model, sen_tokenizer, sentences):
        """Get sentence embeddings."""
        utils.debug_log(
            f"experiments.run_dbandits.LocalHFModelForwardAPI.get_sen_embedding :: Getting embeddings for {len(sentences)} sentences"
        )
        return get_sen_embedding(sen_model, sen_tokenizer, sentences)


def run(
    task,
    n_prompt_tokens,
    nu,
    lamdba,
    n_init,
    n_domain,
    total_iter,
    local_training_iter,
    random_proj,
    intrinsic_dim,
    n_eval,
    gpt,
    init_scale,
    pooling,
    args,
):
    utils.info_log(
        f"experiments.run_dbandits.run :: Starting main run function, task: {task}"
    )
    induce_data, test_data = load_data("induce", task), load_data("eval", task)

    induce_data_size = len(induce_data[0])
    prompt_gen_size = min(int(induce_data_size * 0.5), 100)
    prompt_gen_data, eval_data = data.create_split(induce_data, prompt_gen_size)
    prompt_gen_data = prompt_gen_data[0], [
        random.sample(output, 1)[0] for output in prompt_gen_data[1]
    ]
    demos_template = "Input: [INPUT]\nOutput: [OUTPUT]"
    eval_template = "Instruction: [PROMPT]\n\nInput: [INPUT]\n\nOUTPUT: [OUTPUT]"
    init_prompt = ["\n"]
    prompt_gen_template = "[full_DEMO]\n\nThe instruction was to"

    base_conf = "powhf/configs/instruction_induction.yaml"
    conf = {
        "generation": {
            "num_subsamples": 1,
            "num_demos": 5,
            "num_prompts_per_subsample": 20,
            "model": {"name": "HF_forward", "model_name": gpt},
        },
        "evaluation": {
            "method": "exec_accuracy",
            "task": task,
            "num_samples": min(20, len(eval_data[0])),
            "model": {
                "name": "HF_forward",
                "model_name": gpt,
            },
        },
    }

    def init_qa_gen():
        subsampled_data = data.subsample_data(
            prompt_gen_data, conf["generation"]["num_demos"]
        )
        prompt_gen_template_ = template.InitQATemplate(prompt_gen_template)
        d_template = template.DemosTemplate(demos_template)
        demos = d_template.fill(subsampled_data)
        return prompt_gen_template_.fill(demos)

    model_forward_api = LocalHFModelForwardAPI(
        model_name=gpt,
        eval_data=eval_data,
        init_prompt=init_prompt,
        init_qa_gen=init_qa_gen,
        conf=conf,
        base_conf=base_conf,
        prompt_gen_data=prompt_gen_data,
        n_prompt_tokens=n_prompt_tokens,
        random_proj=random_proj,
        intrinsic_dim=intrinsic_dim,
        magnitude=args.magnitude,
        norm_method=args.norm_method,
    )

    utils.info_log(
        f"experiments.run_dbandits.run :: Setting all seeds, trial: {args.trial}"
    )
    print(set_all_seed(args.trial))
    if args.candidate_method == "induction":
        path_ = f"./query/{task}_{args.n_domain}"
    elif args.candidate_method == "rephrase":
        path_ = f"./query/{task}_{args.n_domain}_rephrase"

    if os.path.exists(path_):
        utils.info_log(
            f"experiments.run_dbandits.run :: Loading initial instructions from file {path_}"
        )
        with open(path_, "r") as fp:
            domains = json.load(fp)
            init_instructions = domains["instructions"]
    else:
        utils.info_log(
            f"experiments.run_dbandits.run :: Generating initial instructions"
        )
        if not os.path.exists("./query"):
            os.mkdir("./query")
        init_instructions = model_forward_api.initialize_prompts(
            args.n_domain, task, args.candidate_method
        )
        with open(path_, "x") as fp:
            domains = {"instructions": init_instructions}
            json.dump(domains, fp, indent=4)

    sen_tokenizer = AutoTokenizer.from_pretrained(
        "sentence-transformers/all-mpnet-base-v2"
    )
    sen_model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")
    sen_embeddings = model_forward_api.get_sen_embedding(
        sen_model, sen_tokenizer, init_instructions
    )
    sen_embeddings = sen_embeddings.to(**tkwargs)

    test_num = 50
    all_tmp_scores = []
    for tmp in range(test_num):
        prompt_tmp = np.random.choice(init_instructions)
        score_tmp = model_forward_api.eval([prompt_tmp], test=True)
        all_tmp_scores += [score_tmp]
    model_forward_api.score_mean = np.mean(all_tmp_scores)
    model_forward_api.score_std = np.std(all_tmp_scores)
    model_forward_api.score_min = np.min(all_tmp_scores)
    model_forward_api.score_max = np.max(all_tmp_scores)
    utils.debug_log(
        f"experiments.run_dbandits.run :: Initial score normalization done, mean: {model_forward_api.score_mean}, std: {model_forward_api.score_std}"
    )
    # select the first m pairs for the first m rounds
    X_train = []
    y_train = []
    select_idx_history = []
    instruction_select_history = []
    for i in range(n_init):
        sen_1_id, sen_2_id = np.random.choice(args.n_domain, 2, replace=False)
        score_1 = model_forward_api.eval([init_instructions[sen_1_id]])
        score_2 = model_forward_api.eval([init_instructions[sen_2_id]])
        instruction_select_history += [
            (init_instructions[sen_1_id], score_1, init_instructions[sen_2_id], score_2)
        ]
        p_ = 1 / (1 + np.exp(-(score_1 - score_2)))
        y_ = np.random.binomial(1, p_)
        X_train += [
            torch.cat(
                [
                    sen_embeddings[sen_1_id].reshape(1, 1, -1),
                    sen_embeddings[sen_2_id].reshape(1, 1, -1),
                ]
            )
        ]
        y_train += [y_]
        select_idx_history += [[sen_1_id, sen_2_id]]

    X_train = torch.cat(X_train, dim=1)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.int32)

    if args.func == "neural":
        l = NeuralDBDiag(
            input_dim=X_train.shape[-1],
            lamdba=lamdba,
            nu=nu,
            init_x=X_train,
            init_y=y_train,
            style="ucb",
            diagonalize=False,
        )
        l.train(local_training_iter=local_training_iter)
    elif args.func == "linear":
        l = LinearDBDiag(
            input_dim=X_train.shape[-1],
            lamdba=lamdba,
            nu=nu,
            init_x=X_train,
            init_y=y_train,
            style="ucb",
            diagonalize=False,
        )
        l.train(local_training_iter=local_training_iter)
    elif args.func == "doublets":
        l = DoubleTS(
            input_dim=X_train.shape[-1],
            lamdba=lamdba,
            nu=nu,
            init_x=X_train,
            init_y=y_train,
            style="ucb",
            diagonalize=False,
        )
        l.train(local_training_iter=local_training_iter)
    elif args.func == "random":
        pass
    max_iter = total_iter - n_init
    utils.info_log(
        f"experiments.run_dbandits.run :: Starting main loop, max_iter: {max_iter}"
    )
    best_r = -np.infty
    best_values = []
    now_values = []
    best_instruction_over_iter = []
    for t in range(max_iter):
        utils.debug_log(
            f"experiments.run_dbandits.run :: Iteration {t}, starting selection"
        )
        if args.func == "random":
            arm_select1, arm_select2 = np.random.choice(args.n_domain, 2, replace=False)
        else:
            arm_select1, arm_select2 = l.select(sen_embeddings, select_idx_history)
            arm_select1, arm_select2 = arm_select1.item(), arm_select2.item()
        select_idx_history += [[arm_select1, arm_select2]]
        score_1 = model_forward_api.eval([init_instructions[arm_select1]])
        score_2 = model_forward_api.eval([init_instructions[arm_select2]])
        instruction_select_history += [
            (
                init_instructions[arm_select1],
                score_1,
                init_instructions[arm_select2],
                score_2,
            )
        ]
        p_ = 1 / (1 + np.exp(-(score_1 - score_2)))
        y_ = np.random.binomial(1, p_)
        if args.func == "random":
            best_arm = arm_select1 if y_ == 1 else arm_select2
        else:
            best_arm = l.find_best(sen_embeddings, select_idx_history)
            best_arm = best_arm.item()

        r = model_forward_api.eval([init_instructions[best_arm]])
        now_values += [r]
        best_instruction_over_iter += [(t, init_instructions[best_arm], r)]

        best_r = max(r, best_r)
        utils.debug_log(
            f"experiments.run_dbandits.run :: Iteration {t}, finished selection, Selected arm: {arm_select1, arm_select2}, reward: {r}"
        )
        new_x_ = torch.cat(
            [
                sen_embeddings[arm_select1].reshape(1, 1, -1),
                sen_embeddings[arm_select2].reshape(1, 1, -1),
            ]
        ).to(**tkwargs)
        if args.func != "random":
            l.train(new_x_, y_, local_training_iter)

        utils.debug_log(
            f"experiments.run_dbandits.run :: Iteration {t}, best reward so far: {best_r}"
        )
        best_values.append(best_r)

    utils.info_log(
        "experiments.run_dbandits.run :: Evaluating best prompt on test data"
    )
    prompts = [best_instruction_over_iter[-1][1]]
    utils.info_log(f"experiments.run_dbandits.run :: Best instruction: {prompts}")

    prompts_set = model_forward_api.return_prompts_set()
    utils.info_log(
        f"experiments.run_dbandits.run :: The final instruction set is: {prompts_set}"
    )
    prompts_list = model_forward_api.return_prompts_list()
    prompts_set = model_forward_api.return_prompts_set()

    test_conf = {
        "generation": {
            "num_subsamples": 3,
            "num_demos": 5,
            "num_prompts_per_subsample": 0,
            "model": {"name": "HF_forward", "model_name": gpt},
        },
        "evaluation": {
            "method": "exec_accuracy",
            "task": task,
            "num_samples": min(100, len(test_data[0])),
            "model": {
                "name": "HF_forward",
                "model_name": gpt,
            },
        },
    }

    test_res, _ = ape.evaluate_prompts(
        prompts=prompts,
        eval_template=eval_template,
        eval_data=test_data,
        few_shot_data=prompt_gen_data,
        demos_template=demos_template,
        conf=test_conf,
        base_conf=base_conf,
    )
    test_res = test_res.sorted()[1][0]
    utils.info_log(
        f"experiments.run_dbandits.run :: Finished main run, test score: {test_res}"
    )
    return (
        test_res,
        prompts,
        prompts_set,
        best_values,
        now_values,
        best_instruction_over_iter,
        init_instructions,
        instruction_select_history,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="InstructZero pipeline")
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="The name of the dataset to use (via the datasets library).",
    )
    parser.add_argument(
        "--n_prompt_tokens", type=int, default=5, help="The number of prompt tokens."
    )
    parser.add_argument("--nu", type=float, default=0.1, help="Set the parameter nu.")
    parser.add_argument(
        "--lamdba", type=float, default=0.1, help="Set the lamdba parameter."
    )
    parser.add_argument(
        "--n_init",
        type=int,
        default=40,
        help="Set the number of initialization points.",
    )
    parser.add_argument(
        "--n_domain", type=int, default=500, help="Set the number of domain."
    )
    parser.add_argument(
        "--total_iter", type=int, default=165, help="Set the number of total queries."
    )
    parser.add_argument(
        "--local_training_iter",
        type=int,
        default=30,
        help="Set the number of total queries.",
    )
    parser.add_argument(
        "--random_proj", type=str, default="uniform", help="Set the projection method."
    )
    parser.add_argument(
        "--intrinsic_dim",
        type=int,
        default=100,
        help="Set the number of intrinsic dim.",
    )
    parser.add_argument(
        "--n_eval",
        type=int,
        default=1000,
        help="Set the number of domains to be evaluated at each ucb iteration.",
    )
    parser.add_argument(
        "--name", type=str, default="", help="Set the name of the experiments."
    )
    parser.add_argument(
        "--gpt",
        type=str,
        default="google/flan-t5-base",
        help="Which version of gpt to use.",
    )
    parser.add_argument(
        "--init_scale", type=float, default=1, help="Which scale to use."
    )
    parser.add_argument(
        "--pooling", type=str, default="last", help="Which pooling method to use."
    )
    parser.add_argument(
        "--func",
        type=str,
        default="neural",
        help="Which model to use, can be linear, neural.",
    )
    parser.add_argument("--trial", type=int, default=0, help="Trial ID.")
    parser.add_argument(
        "--magnitude", type=int, default=10, help="The magnitude of the scores."
    )
    parser.add_argument(
        "--norm_method",
        type=str,
        default="standard",
        help="The way to transform the value, standard, minmax.",
    )
    parser.add_argument(
        "--candidate_method",
        type=str,
        default="induction",
        help="The way to generate candidates.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    utils.info_log(
        f"experiments.run_dbandits.main :: Starting main execution, args: {args}"
    )
    print(set_all_seed(0))
    (
        test_score,
        prompts,
        prompts_set,
        best_values,
        now_values,
        best_instruction_over_iter,
        init_instructions,
        instruction_select_history,
    ) = run(
        task=args.task,
        n_prompt_tokens=args.n_prompt_tokens,
        nu=args.nu,
        lamdba=args.lamdba,
        n_init=args.n_init,
        n_domain=args.n_domain,
        total_iter=args.total_iter,
        local_training_iter=args.local_training_iter,
        random_proj=args.random_proj,
        intrinsic_dim=args.intrinsic_dim,
        n_eval=args.n_eval,
        gpt=args.gpt,
        init_scale=args.init_scale,
        pooling=args.pooling,
        args=args,
    )

    args_dict = vars(args)
    args_dict["test_score"] = test_score
    args_dict["valid_score"] = best_values[-1]
    args_dict["best_prompt"] = prompts
    args_dict["prompts_set"] = prompts_set
    args_dict["best_values"] = best_values
    args_dict["best_instruction_over_iter"] = best_instruction_over_iter
    args_dict["init_instructions"] = init_instructions
    args_dict["instruction_select_history"] = instruction_select_history
    args_dict["now_values"] = now_values

    save_dir = "./results/" + args.name

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    path = os.path.join(
        save_dir,
        args.task
        + datetime.now().strftime("-%Y-%m-%d_%H-%M-%S")
        + "_trial{}".format(args.trial)
        + ".json",
    )
    utils.info_log(f"experiments.run_dbandits.main :: Saving results to {path}")
    with open(path, "x") as fp:
        json.dump(args_dict, fp, indent=4)

    print("Finished!!!")
    print(f"Test score on HF model: {test_score}")
    utils.info_log("experiments.run_dbandits.main :: Finished main execution")
