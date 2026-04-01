import logging

from .registry import register_model
from .llama2 import create_tokenizer, create_embed_model
from .llama2 import create_model as create_llama2_model
from transformers import AutoTokenizer

TOKENIZER_ARGS = dict(model_max_length=8192)


__HF_MODEL_MAP = {
    "8b": "Meta-Llama-3.1-8B",
    "8b-instruct": "Meta-Llama-3.1-8B-Instruct",
    "70b": "Meta-Llama-3.1-70B",
    "70b-instruct": "Meta-Llama-3.1-70B-Instruct",
    "405b": "Meta-Llama-3.1-405B",
    "405b-instruct": "Meta-Llama-3.1-405B-Instruct",
}


def __get_model_hf_id(model_str):
    try:
        _, kind = model_str.split(":")

        assert kind in __HF_MODEL_MAP.keys()
    except ValueError:
        logging.exception(
            f'Model string should be formatted as "llama3_1:<kind>" (Got {model_str})',
        )
        raise
    except AssertionError:
        logging.exception(
            f'Model not found. Model string should be formatted as "llama3_1:<kind>" (Got {model_str})',
        )
        raise

    return __HF_MODEL_MAP[kind]


def create_tokenizer_and_model(kind, tokenizer_args=None, **kwargs):
    tokenizer = create_tokenizer(kind, **(tokenizer_args or dict()))
    model = create_llama2_model(kind, tokenizer=tokenizer, **kwargs)
    return tokenizer, model


@register_model(**TOKENIZER_ARGS)
def llama3_1_tokenizer(*, model_str=None, **kwargs):
    return create_tokenizer(__get_model_hf_id(model_str), **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_1(*, model_str=None, **kwargs):
    return create_tokenizer_and_model(__get_model_hf_id(model_str), **kwargs)


@register_model(tokenizer_args=TOKENIZER_ARGS)
def llama3_1_embed(*, model_str=None, **kwargs):
    return create_embed_model(__get_model_hf_id(model_str), **kwargs)
