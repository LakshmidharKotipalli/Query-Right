import streamlit as st

from routing.router import route_query


def render_citations(citations, route):
    """Render source citations in an expandable section."""
    label = "Web Sources" if route == "WEB" else "Document Sources"
    with st.expander(f"Sources ({len(citations)} {label.lower()})"):
        for i, cite in enumerate(citations, 1):
            if route == "WEB":
                st.markdown(f"**{i}. [{cite['source']}]({cite['page']})**")
            else:
                st.markdown(f"**{i}. {cite['source']}** (Page {cite['page']})")
            st.caption(cite.get("preview", ""))
            st.divider()


def render_chat():
    """Render the main chat interface."""
    st.header("Legal Document Q&A")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "citations" in message and message["citations"]:
                render_citations(message["citations"], message.get("route", "LOCAL"))

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = route_query(prompt)

            st.markdown(result["answer"])

            if result["citations"]:
                render_citations(result["citations"], result["route"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "citations": result["citations"],
            "route": result["route"],
        })
