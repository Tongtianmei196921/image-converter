from biomni.agent.a1 import (
    _gradio_composer_html,
    _gradio_empty_prompt_payload,
    _gradio_failure_message,
    _gradio_header_html,
    _gradio_guess_llm_provider,
    _gradio_llm_provider_defaults,
    _gradio_llm_provider_form_state,
    _gradio_llm_provider_presets,
    _gradio_locale_copy,
    _gradio_match_llm_provider,
    _gradio_prompt_payload,
    _gradio_prompt_presets,
    _gradio_resolve_runtime_llm_settings,
    _gradio_quickstart_html,
    _gradio_run_guide_html,
    _gradio_status_pill_html,
    _gradio_timeout_message,
    _gradio_workbench_css,
    _gradio_working_message,
)


def test_gradio_workbench_css_contains_core_shell_selectors():
    css = _gradio_workbench_css()

    assert "--p-bg" in css
    assert "#biomni-shell" in css
    assert ".biomni-panel" in css
    assert "#biomni-composer" in css
    assert "@media (max-width: 900px)" in css
    assert "pointer-events: auto !important;" in css
    assert "#biomni-language-selector" in css


def test_gradio_header_html_contains_branding_copy():
    header = _gradio_header_html()

    assert "Biomni" in header
    assert "Research Workbench" in header
    assert "Run biomedical workflows" in header


def test_gradio_locale_copy_exposes_chinese_first_labels():
    copy = _gradio_locale_copy("zh")

    assert copy["language_label"] == "界面语言"
    assert copy["header_title"] == "研究工作台"
    assert copy["model_routing_title"] == "模型路由"


def test_gradio_locale_copy_falls_back_to_english():
    copy = _gradio_locale_copy("fr")

    assert copy["language_label"] == "Language"
    assert copy["header_title"] == "Research Workbench"


def test_gradio_header_html_can_render_chinese_copy():
    header = _gradio_header_html("zh")

    assert "研究工作台" in header
    assert "生物医学" in header


def test_gradio_composer_html_can_render_chinese_copy():
    html = _gradio_composer_html("zh")

    assert "Shift+Enter 换行" in html
    assert "biomni-composer-hint" in html


def test_gradio_status_pill_html_renders_label_and_value():
    pill = _gradio_status_pill_html("Model", "LongCat-Flash-Thinking-2601")

    assert "Model" in pill
    assert "LongCat-Flash-Thinking-2601" in pill
    assert "biomni-status-pill" in pill


def test_gradio_quickstart_html_contains_research_tracks():
    html = _gradio_quickstart_html()

    assert "Quick Starts" in html
    assert "Drug Label Review" in html
    assert "Literature Triage" in html
    assert "Single-Cell Readout" in html


def test_gradio_quickstart_html_can_render_chinese_copy():
    html = _gradio_quickstart_html("zh")

    assert "快速开始" in html
    assert "药品标签审阅" in html
    assert "文献分诊" in html
    assert "单细胞解读" in html


def test_gradio_run_guide_html_contains_split_view_copy():
    html = _gradio_run_guide_html()

    assert "Run Discipline" in html
    assert "Console" in html
    assert "Trace" in html


def test_gradio_run_guide_html_can_render_chinese_copy():
    html = _gradio_run_guide_html("zh")

    assert "运行准则" in html
    assert "结果 Console" in html
    assert "执行 Trace" in html


def test_gradio_prompt_presets_expose_expected_quick_actions():
    presets = _gradio_prompt_presets()

    assert "Drug Label Review" in presets
    assert "Literature Triage" in presets
    assert "Single-Cell Readout" in presets
    assert "FDA" in presets["Drug Label Review"]


def test_gradio_prompt_payload_returns_multimodal_shape():
    payload = _gradio_prompt_payload("Drug Label Review")

    assert payload["text"]
    assert payload["files"] == []
    assert "FDA" in payload["text"]


def test_gradio_prompt_payload_can_render_chinese_prompt_copy():
    payload = _gradio_prompt_payload("Drug Label Review", "zh")

    assert payload["files"] == []
    assert "FDA" in payload["text"]
    assert "说明书" in payload["text"]


def test_gradio_empty_prompt_payload_clears_text_and_files():
    payload = _gradio_empty_prompt_payload()

    assert payload == {"text": "", "files": []}


def test_gradio_working_message_is_product_copy():
    message = _gradio_working_message()

    assert message == "Biomni is working through the request."


def test_gradio_timeout_message_mentions_elapsed_timeout():
    message = _gradio_timeout_message(45)

    assert "45" in message
    assert "did not report progress" in message


def test_gradio_failure_message_surfaces_exception_text():
    message = _gradio_failure_message(RuntimeError("upstream stalled"))

    assert "upstream stalled" in message
    assert "execution error" in message


def test_gradio_llm_provider_presets_include_zhipu_official_endpoint():
    presets = _gradio_llm_provider_presets()

    assert "Zhipu" in presets
    assert presets["Zhipu"]["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"
    assert presets["Zhipu"]["default_model"] == "glm-5.1"
    assert "glm-4.5" in presets["Zhipu"]["models"]


def test_gradio_llm_provider_presets_include_deepseek_official_endpoint():
    presets = _gradio_llm_provider_presets()

    assert "DeepSeek" in presets
    assert presets["DeepSeek"]["base_url"] == "https://api.deepseek.com"
    assert presets["DeepSeek"]["default_model"] == "deepseek-v4-flash"
    assert "deepseek-v4-pro" in presets["DeepSeek"]["models"]


def test_gradio_match_llm_provider_identifies_known_profiles():
    assert _gradio_match_llm_provider("glm-4.5", "https://open.bigmodel.cn/api/paas/v4/") == "Zhipu"
    assert _gradio_match_llm_provider(
        "LongCat-Flash-Thinking-2601",
        "https://api.longcat.chat/openai",
    ) == "LongCat"
    assert _gradio_match_llm_provider("deepseek-v4-flash", "https://api.deepseek.com") == "DeepSeek"
    assert _gradio_match_llm_provider("unknown-model", "https://example.com/v1") == "Custom"


def test_gradio_llm_provider_defaults_switch_to_selected_profile():
    defaults = _gradio_llm_provider_defaults(
        "Zhipu",
        current_model="LongCat-Flash-Thinking-2601",
        current_base_url="https://api.longcat.chat/openai",
    )

    assert defaults["provider"] == "Zhipu"
    assert defaults["model"] == "glm-5.1"
    assert defaults["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"


def test_gradio_llm_provider_defaults_preserve_active_model_for_same_provider():
    defaults = _gradio_llm_provider_defaults(
        "Zhipu",
        current_model="glm-5.1",
        current_base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    assert defaults["provider"] == "Zhipu"
    assert defaults["model"] == "glm-5.1"
    assert defaults["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"


def test_gradio_llm_provider_form_state_exposes_model_choices():
    state = _gradio_llm_provider_form_state(
        "Zhipu",
        current_model="glm-5.1",
        current_base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    assert state["provider"] == "Zhipu"
    assert state["model"] == "glm-5.1"
    assert state["base_url"] == "https://open.bigmodel.cn/api/paas/v4/"
    assert state["model_choices"][0] == "glm-5.1"
    assert "glm-4.7" in state["model_choices"]


def test_gradio_llm_provider_form_state_keeps_custom_model_when_provider_matches():
    state = _gradio_llm_provider_form_state(
        "Zhipu",
        current_model="glm-5",
        current_base_url="https://open.bigmodel.cn/api/paas/v4/",
    )

    assert state["model"] == "glm-5"
    assert "glm-5.1" in state["model_choices"]


def test_gradio_llm_provider_form_state_clears_choices_for_custom_provider():
    state = _gradio_llm_provider_form_state(
        "Custom",
        current_model="lab-model",
        current_base_url="https://lab.example/v1",
    )

    assert state["provider"] == "Custom"
    assert state["model"] == "lab-model"
    assert state["base_url"] == "https://lab.example/v1"
    assert state["model_choices"] == []


def test_gradio_llm_provider_defaults_preserve_custom_inputs():
    defaults = _gradio_llm_provider_defaults(
        "Custom",
        current_model="my-lab-model",
        current_base_url="https://lab.example/v1",
    )

    assert defaults["provider"] == "Custom"
    assert defaults["model"] == "my-lab-model"
    assert defaults["base_url"] == "https://lab.example/v1"


def test_gradio_guess_llm_provider_recognizes_deepseek_aliases():
    assert _gradio_guess_llm_provider("deepseek") == "DeepSeek"
    assert _gradio_guess_llm_provider("deepseek-v4-pro") == "DeepSeek"
    assert _gradio_guess_llm_provider("glm-5.1") == "Zhipu"


def test_gradio_resolve_runtime_llm_settings_autofills_deepseek_provider():
    resolved = _gradio_resolve_runtime_llm_settings(
        "Custom",
        model_name="deepseek",
        base_url="",
    )

    assert resolved["provider"] == "DeepSeek"
    assert resolved["model"] == "deepseek-v4-flash"
    assert resolved["base_url"] == "https://api.deepseek.com"


def test_gradio_llm_provider_form_state_promotes_known_custom_deepseek_models():
    state = _gradio_llm_provider_form_state(
        "Custom",
        current_model="deepseek-v4-pro",
        current_base_url="",
    )

    assert state["provider"] == "DeepSeek"
    assert state["model"] == "deepseek-v4-pro"
    assert state["base_url"] == "https://api.deepseek.com"
    assert "deepseek-v4-flash" in state["model_choices"]
