"""Contains classes for querying local Hugging Face language models."""

import os
import time
from tqdm import tqdm
from abc import ABC, abstractmethod
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
from powhf import utils


def model_from_config(config, disable_tqdm=True):
    """Returns an LLM based on the config."""
    utils.debug_log(f"powhf.llm.model_from_config :: Loading model: {config['name']}")
    model_type = config["name"]
    if model_type == "HF_forward":
        return HF_Forward(config, disable_tqdm=disable_tqdm)
    elif model_type == "HF_insert":
        return HF_Insert(config, disable_tqdm=disable_tqdm)
    else:
        utils.debug_log(
            f"powhf.llm.model_from_config :: Unknown model type: {model_type}"
        )
        raise ValueError(f"Unknown model type: {model_type}")


class LLM(ABC):
    """Abstract base class for LLMs."""

    @abstractmethod
    def generate_text(self, prompt):
        pass

    @abstractmethod
    def log_probs(self, text, log_prob_range):
        pass


class HF_Forward(LLM):
    """Wrapper for Hugging Face models."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the HF model."""
        utils.debug_log(
            f"powhf.llm.HF_Forward :: Initializing model: {config['model_name']}"
        )
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float16
        ).to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def generate_text(self, prompts, n, temperature=0, use_seed=True):
        """Generates text from the local HF model."""
        utils.debug_log(
            f"powhf.llm.HF_Forward.generate_text :: Generating text, num prompts: {len(prompts)}, n: {n}"
        )
        if not isinstance(prompts, list):
            prompts = [prompts]

        batch_size = 10
        text = []
        start_time = time.time()
        for i in range(len(prompts) // batch_size + 1):
            tmp_prompts = prompts[i * batch_size : (i + 1) * batch_size]
            if len(tmp_prompts) > 0:
                input_ids = self.tokenizer(
                    tmp_prompts, padding="longest", return_tensors="pt"
                ).input_ids.to(device=self.device)
                outputs = self.model.generate(
                    input_ids, max_new_tokens=32, temperature=temperature
                )
                text += self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        end_time = time.time()
        utils.debug_log(
            f"powhf.llm.HF_Forward.generate_text :: Generation time: {end_time-start_time:.4f} seconds"
        )
        return text

    def log_probs(self, text, log_prob_range=None):
        utils.debug_log(
            "powhf.llm.HF_Forward.log_probs :: Log probs not implemented for HF models."
        )
        raise NotImplementedError


class HF_Insert(LLM):
    """Wrapper for Hugging Face models."""

    def __init__(self, config, needs_confirmation=False, disable_tqdm=True):
        """Initializes the HF model."""
        utils.debug_log(
            f"powhf.llm.HF_Insert :: Initializing model: {config['model_name']}"
        )
        self.config = config
        self.needs_confirmation = needs_confirmation
        self.disable_tqdm = disable_tqdm
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda:0"
        else:
            self.device = "cpu"

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            config["model_name"], torch_dtype=torch.float16
        ).to(device=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    def generate_text(self, prompts, n, temperature=0, use_seed=True):
        """Generates text from the local HF model with an insertion prefix and suffix."""
        utils.debug_log(
            f"powhf.llm.HF_Insert.generate_text :: Generating text, num prompts: {len(prompts)}, n: {n}"
        )
        if not isinstance(prompts, list):
            prompts = [prompts]
        batch_size = self.config["batch_size"]
        assert batch_size == 1  # Ensure that batch_size is 1
        text = []
        start_time = time.time()
        for prompt in prompts:
            prefix = prompt.split("[APE]")[0]
            suffix = prompt.split("[APE]")[1]
            input_ids = self.tokenizer(prefix, return_tensors="pt").input_ids.to(
                device=self.device
            )
            outputs = self.model.generate(
                input_ids, max_new_tokens=32, temperature=temperature
            )
            output = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            text.append(
                output + suffix
            )  # Add the suffix back in since it is not generated by the model
        end_time = time.time()
        utils.debug_log(
            f"powhf.llm.HF_Insert.generate_text :: Generation time: {end_time-start_time:.4f} seconds"
        )
        return text

    def log_probs(self, text, log_prob_range=None):
        utils.debug_log(
            "powhf.llm.HF_Insert.log_probs :: Log probs not implemented for HF models."
        )
        raise NotImplementedError
