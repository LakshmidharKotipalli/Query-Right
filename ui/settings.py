import subprocess

import streamlit as st

import config


def _get_ollama_models() -> list[str]:
    """Fetch locally available Ollama models."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=5,
        )
        models = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            name = line.split()[0]
            if name:
                models.append(name)
        return models if models else [config.LLM_MODEL]
    except Exception:
        return [config.LLM_MODEL]


def render_settings_button():
    """Render a settings gear button pinned to the top-right corner."""
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = config.LLM_MODEL

    models = _get_ollama_models()

    # Ensure current selection is in the list
    if st.session_state.selected_model not in models:
        models.insert(0, st.session_state.selected_model)

    # Top-right positioned settings using columns
    _, right = st.columns([9, 1])
    with right:
        with st.popover("Settings", use_container_width=True):
            st.markdown("**Model**")
            selected = st.selectbox(
                "Ollama model",
                options=models,
                index=models.index(st.session_state.selected_model),
                key="model_selector",
                label_visibility="collapsed",
            )

            st.markdown("**Temperature**")
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=config.LLM_TEMPERATURE,
                step=0.1,
                key="temp_slider",
                label_visibility="collapsed",
                help="Lower = more factual, Higher = more creative",
            )

            st.markdown("**Context Window**")
            num_ctx = st.select_slider(
                "Context window",
                options=[2048, 4096, 8192, 16384, 32768],
                value=config.LLM_NUM_CTX,
                key="ctx_slider",
                label_visibility="collapsed",
            )

            # Apply changes
            if selected != st.session_state.selected_model:
                st.session_state.selected_model = selected
            config.LLM_MODEL = st.session_state.selected_model
            config.LLM_TEMPERATURE = temperature
            config.LLM_NUM_CTX = num_ctx
