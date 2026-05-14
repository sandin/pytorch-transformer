import random
from pathlib import Path
from html import escape

import gradio as gr

from config import DEFAULT_CONFIG_PATH, get_config, load_translation_dataset
from dataset import get_sentence
from translate import Translator


DEFAULT_TITLE = "Transformer Translator"
DEFAULT_DESCRIPTION = "A machine translation chatbot built with the Transformer architecture."
DEFAULT_EXAMPLES = [
    ["Who are you?"],
    ["Work you've done here is outstanding."],
]
EXAMPLE_COUNT = 3
TRANSLATORS = {}


def list_config_files():
    config_files = sorted(Path("configs").glob("*.json"))
    return [str(config_file) for config_file in config_files]


def default_config_path(config_files):
    default_path = str(DEFAULT_CONFIG_PATH)
    if default_path in config_files:
        return default_path
    return config_files[0] if config_files else default_path


def get_config_info(config_path: str):
    try:
        config = get_config(Path(config_path or DEFAULT_CONFIG_PATH))
    except Exception:
        return "", "", ""

    return config.get("datasource", ""), config.get("lang_src", ""), config.get("lang_tgt", "")


def get_language_hint(config_path: str):
    datasource, src_lang, tgt_lang = get_config_info(config_path)
    return (
        "<div style='color:#8a8f98;font-size:12px;line-height:1.4;"
        "margin-top:-10px;opacity:.78'>"
        f"Dataset <span style='color:#737983'>{datasource}</span>"
        " · "
        f"Source <span style='color:#737983'>{src_lang}</span>"
        " · "
        f"Target <span style='color:#737983'>{tgt_lang}</span>"
        "</div>"
    )


def get_random_examples(config_path: str, count: int = EXAMPLE_COUNT):
    try:
        config = get_config(Path(config_path or DEFAULT_CONFIG_PATH))
        ds = load_translation_dataset(config, split="all")
        sample_count = min(count, len(ds))
        indexes = random.sample(range(len(ds)), sample_count)
        return [[get_sentence(ds[index], config["lang_src"])] for index in indexes]
    except Exception as exc:
        print(f"Failed to load random examples: {exc}")
        return DEFAULT_EXAMPLES


def get_translator(config_path: str):
    config = str(Path(config_path or DEFAULT_CONFIG_PATH))
    if config not in TRANSLATORS:
        TRANSLATORS[config] = Translator(config)
    return TRANSLATORS[config]


def respond(message: str, history, config_path: str):
    """Translate a single chat message for Gradio's ChatInterface."""
    sentence = (message or "").strip()
    if not sentence:
        return "Please enter a sentence to translate."

    try:
        translator = get_translator(config_path)
        result = translator.translate(sentence, return_stats=True)
        return (
            f"{escape(result.text)}"
            "<div style='color:#8a8f98;font-size:12px;line-height:1.4;"
            "margin-top:8px;opacity:.78'>"
            f"{result.tokens_per_second:.2f} tok/s"
            "</div>"
        )
    except Exception as exc:
        return f"Translation failed: {exc}"


def build_ui():
    config_files = list_config_files()
    selected_config = default_config_path(config_files)
    examples = get_random_examples(selected_config)

    with gr.Blocks(title=DEFAULT_TITLE) as demo:
        config_input = gr.Dropdown(
            choices=config_files,
            value=selected_config,
            label="Config",
            interactive=True,
        )
        language_hint = gr.HTML(value=get_language_hint(selected_config))

        config_input.change(
            fn=get_language_hint,
            inputs=config_input,
            outputs=language_hint,
        )

        gr.ChatInterface(
            fn=respond,
            chatbot=gr.Chatbot(
                label="AI Translator",
                show_label=True,
            ),
            title=DEFAULT_TITLE,
            description=DEFAULT_DESCRIPTION,
            additional_inputs=[config_input],
            additional_inputs_accordion="Settings",
            textbox=gr.Textbox(
                placeholder="Enter a sentence to translate, then press Enter",
                show_label=False,
                container=True,
                lines=1,
                max_lines=1,
                scale=7,
            ),
            examples=examples,
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
