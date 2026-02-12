import streamlit as st

from ui.sidebar import render_sidebar
from ui.chat import render_chat
from ui.settings import render_settings_button

st.set_page_config(
    page_title="Query Right - Legal Q&A",
    page_icon="gavel",
    layout="wide",
    initial_sidebar_state="expanded",
)


def main():
    render_sidebar()
    render_settings_button()
    render_chat()


if __name__ == "__main__":
    main()
