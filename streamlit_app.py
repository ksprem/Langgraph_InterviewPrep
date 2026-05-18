import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="LangGraph Interview Prep",
    page_icon="🚀",
    layout="centered",
)

st.title("LangGraph Interview Prep Assistant")
st.markdown(
    "Use this app to generate a personalized interview preparation plan based on your role, urgency, and confidence level."
)

st.markdown(
    "### How it works\n" \
    "1. Enter a short description of the interview you're preparing for.\n" \
    "2. The app uses LangGraph and OpenAI to create technical, behavioral, and confidence suggestions.\n" \
    "3. You'll get either a quick or deep practice plan with a message log."
)

user_input = st.text_area(
    "Describe your interview preparation needs:",
    placeholder="I have a backend engineer interview in two days and I feel underprepared.",
    height=180,
)

if st.button("Generate Plan"):
    if not user_input.strip():
        st.warning("Please enter your interview description first.")
    else:
        with st.spinner("Generating your personalized prep plan..."):
            try:
                from interview_prep_graph import app

                result = app.invoke({
                    "user_input": user_input,
                    "messages": [],
                })

                st.subheader("Personalized Preparation Plan")
                st.code(result["final_plan"], language="text")

                st.subheader("Decision Log")
                st.write(
                    "This plan was selected based on the interview details and the specialist suggestions."
                )
                st.markdown(f"**Reason:** {result.get('prep_reason', 'N/A')}  ")

                if result.get("messages"):
                    st.subheader("Message Log")
                    for msg in result["messages"]:
                        st.write(f"- {msg}")
            except Exception as exc:
                st.error("Something went wrong while generating the plan.")
                st.exception(exc)

st.markdown("---")
st.markdown(
    "**Note:** The app requires an OpenAI API key configured as `OPENAI_API_KEY` in the environment. "
    "On Railway, set this value in the project settings."
)
