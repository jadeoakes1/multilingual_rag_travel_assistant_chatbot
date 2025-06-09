# Streamlit app for multilingual travel assistant chatbot with optional sampling

import streamlit as st
from multilingual_rag_chatbot_llm import generate_response, format_prompt

st.set_page_config(page_title="Multilingual Travel Assistant Chatbot", layout="centered")
st.title("Multilingual Travel Assistant Chatbot")
st.markdown("Ask about language, grammar, or travel info in English or Spanish")

# Input and settings
lang = st.selectbox("Select language:", ["en", "es"])

# Example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("Example: General (English)"):
        st.session_state["example_query"] = "Is this sentence correct: 'She don't like apples'?"
with col2:
    if st.button("Example: Travel (Spanish)"):
        st.session_state["example_query"] = "¿Cuáles son 3 cosas que puedo hacer en Oaxaca?"

user_input = st.text_area("Enter your message:", height=100, value=st.session_state.get("example_query", ""))
mode = st.selectbox("Select mode:", ["general", "travel", "no_retrieval"])
do_sample = st.checkbox("Use sampling (creative output)?", value=False)

if do_sample:
    top_p = st.slider("Top-p (nucleus sampling)", 0.1, 1.0, 0.85)
    temperature = st.slider("Temperature", 0.1, 1.5, 0.6)
else:
    top_p = temperature = None

# Handle response generation
if st.button("Send"):
    if not user_input.strip():
        st.warning("Please enter a message")
    else:
        with st.spinner("Generating response..."):
                answer = generate_response(
                    user_input=user_input,
                    mode=mode,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature
                )

        st.success("Bot Response:")
        st.markdown(answer)

        # Show retrieved context (except in no_retrieval mode)
        if mode != "no_retrieval":
            from multilingual_rag_chatbot_llm import retrieve_context
            context = retrieve_context(user_input, k=5, source=mode)

            st.markdown("**Context Used:**")
            for c in context:
                if mode == "travel":
                    st.markdown(f"- {c['text']}")
                else:
                    st.markdown(f"- {c.get('en', '')} → {c.get('es', '')}")
        else:
            context = []

        # Build full prompt for download
        full_prompt = format_prompt(user_input, context if mode != "no_retrieval" else [], source_mode=mode)
        st.download_button(
            label="Download Prompt + Answer",
            data=f"{full_prompt}\n\nAnswer:\n{answer}",
            file_name="chatbot_output.txt"
        )
