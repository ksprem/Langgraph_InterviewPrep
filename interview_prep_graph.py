# =============================================================================
# Interview Prep Suggester -- A LangGraph Learning Project
# =============================================================================
#
# This project teaches you how LangGraph works by building an interview preparation
# assistant that suggests personalized prep plans.
#
# WHAT THIS DOES:
# A user enters information about the interview they are preparing for (e.g. "I have an interview for a software engineer position tomorrow"). The system runs 3 suggestion engines in
# PARALLEL (technical, behavioural, confidence), then a decision node picks the
# best approach and routes to either a QUICK practice (under 5 minutes) or a
# DEEPER session (10-15 minutes) based on severity.
#
# LANGGRAPH CONCEPTS COVERED:
# 1. State Management (Pydantic) -- user input flows through the graph
# 2. Nodes -- each function does one job (suggest technical, behavioural, confidence)
# 3. Parallel Execution -- 3 suggestion nodes run at the same time
# 4. Fan-in -- waiting for all 3 suggestions before picking the best
# 5. Conditional Edges -- routing to quick vs deep based on severity
# 6. Graph Compilation -- turning the graph definition into a runnable app
#
# GRAPH STRUCTURE:
#
#   START
#     |
#   understand_interview_context
#     |
#     +---> suggest_technical --------+
#     |                               |
#     +---> suggest_behavioural ------+---> pick_best_practice
#     |                               |         |
#     +---> suggest_confidence ---------+    (conditional)
#                                        /          \
#                                   quick?         deep?
#                                     |               |
#                               quick_practice   deep_practice
#                                     |               |
#                                    END             END
#
# HOW TO RUN:
#   python interview_prep_graph.py
#
# DEPENDENCIES (same as requirements.txt):
#   langgraph, langchain-openai, python-dotenv, pydantic
#
# =============================================================================

import sys
import operator
import json
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()


class InterviewPrepState(BaseModel):
    user_input: str = ""
    technical_suggestion: str = ""
    behavioural_suggestion: str = ""
    confidence_suggestion: str = ""
    job_role: str = ""
    urgency_level: str = ""
    preparedness_level: str = ""
    needs_deep_prep: bool = False
    prep_reason: str = ""
    final_plan: str = ""
    messages: Annotated[list, operator.add] = []


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


def understand_interview_context(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are an interview preparation intake assistant.\n\n"
        f"The user says:\n'{state.user_input}'\n\n"
        f"Extract the following fields:\n"
        f"1. job_role: the role they are interviewing for\n"
        f"2. urgency_level: LOW, MEDIUM, or HIGH\n"
        f"   - HIGH means interview is today/tomorrow/very soon or user sounds very stressed\n"
        f"   - MEDIUM means interview is in a few days or user feels somewhat prepared\n"
        f"   - LOW means interview is later or user has enough time\n"
        f"3. preparedness_level: CONFIDENT, MODERATE, or UNDERPREPARED\n\n"
        f"Reply STRICTLY in this JSON format with no extra text:\n"
        f'{{"job_role": "role name", "urgency_level": "LOW/MEDIUM/HIGH", '
        f'"preparedness_level": "CONFIDENT/MODERATE/UNDERPREPARED"}}'
    )
    return {
        "messages": [f"[understand_interview_context] {response.content}"]
    }


def technical_suggestion(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a technical interview preparation specialist.\n\n"
        f"User input: '{state.user_input}'\n"
        f"Target role: {state.job_role}\n"
        f"Urgency level: {state.urgency_level}\n"
        f"Preparedness level: {state.preparedness_level}\n\n"
        f"Suggest the most important technical topics to review for this interview.\n"
        f"Focus on data structures, system design, not a huge syllabus.\n"
        f"Include:\n"
        f"- Top technical concepts\n"
        f"- One or two hands-on practice tasks\n"
        f"- Common mistakes to avoid\n\n"
        f"Keep it practical and concise."
    )
    return {
        "technical_suggestion": response.content,
        "messages": [f"[technical_suggestion] Done"]
    }


def behavioural_suggestion(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a behavioral interview coach.\n\n"
        f"User input: '{state.user_input}'\n"
        f"Target role: {state.job_role}\n"
        f"Urgency level: {state.urgency_level}\n"
        f"Preparedness level: {state.preparedness_level}\n\n"
        f"Suggest behavioral story prompts for this role using the STAR method and Past wins.\n"
        f"Include: \n"
        f"- 4 to 5 likely behavioral questions\n"
        f"- What kind of real example the candidate should prepare\n"
        f"- How to structure each answer briefly using STAR\n\n"
        f"Keep it concise and interview-focused."
    )
    return {
        "behavioural_suggestion": response.content,
        "messages": [f"[behavioural_suggestion] Done"]
    }


def confidence_suggestion(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a confident Interview Preparer and always success oriented. "
        f"List out all Confidence building habits  '{state.user_input}'. "
        f"Include: \n"
        f"- Power Pose \n"
        f"- Breathing before the interview call \n"
        f"- Make this person feel like they are going to CRUSH this interview and get the job offer! "
    )
    return {
        "confidence_suggestion": response.content,
        "messages": [f"[confidence_suggestion] Done"]
    }


def pick_best_prep(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a Interview preparation decision system. The user feels: '{state.user_input}'.\n\n"
        f"Here are three suggestions from specialists:\n\n"
        f"TECHNICAL:\n{state.technical_suggestion}\n\n"
        f"BEHAVIOURAL:\n{state.behavioural_suggestion}\n\n"
        f"CONFIDENCE:\n{state.confidence_suggestion}\n\n"
        f"Decide: does this person need a QUICK PRACTICE (1-hour focused drill — just the top 3 gaps) "
        f"or a DEEP PRACTICE (a structured 3-hour study plan with timed blocks for each area)?\n\n"
        f"Reply STRICTLY in this JSON format (no other text):\n"
        f'{{"needs_deep_prep": true/false, "reason": "one sentence explanation"}}'
    )
    try:
        result = json.loads(response.content)
        needs_deep_prep = result["needs_deep_prep"]
        reason = result["reason"]
    except (json.JSONDecodeError, KeyError):
        needs_deep_prep = False
        reason = "Could not parse decision, defaulting to quick practice."

    return {
        "needs_deep_prep": needs_deep_prep,
        "prep_reason": reason,
        "messages": [f"[pick_best_practice] deep_session={needs_deep_prep}, prep_reason={reason}"]
    }


def quick_prep(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a expert Interview Preparer. The user feels: '{state.user_input}'.\n\n"
        f"Based on these specialist suggestions, create a SHORT practice (under 1 hour focused plan) "
        f"that combines the best elements:\n\n"
        f"TECHNICAL:\n{state.technical_suggestion}\n\n"
        f"BEHAVIOURAL:\n{state.behavioural_suggestion}\n\n"
        f"CONFIDENCE:\n{state.confidence_suggestion}\n\n"
        f"Format it as a simple numbered list of steps. "
        f"Keep it warm, encouraging, and easy to follow. End with a kind closing line."
    )
    return {
        "final_plan": f"QUICK  PRACTICE (under 1 hour)\n{'='*45}\n{response.content}",
        "messages": [f"[quick_practice] Generated quick practice"]
    }


def deep_prep(state: InterviewPrepState) -> dict:
    response = llm.invoke(
        f"You are a compassionate wellness coach. The user feels: '{state.user_input}'.\n\n"
        f"Based on these specialist suggestions, create a DEEPER session (3 hour study Plan) "
        f"that thoughtfully combines all three approaches:\n\n"
        f"TECHNICAL: {state.technical_suggestion}\n"
        f"BEHAVIOURAL: {state.behavioural_suggestion}\n"
        f"CONFIDENCE: {state.confidence_suggestion}\n\n"
        f"Structure it in 3 phases: Technical (data structures, system design), Behavioral (STAR method, past wins), Confidence (power pose, breathing before the call). "
        f"Give clear step-by-step instructions for each phase with timing. "
        f"Keep it warm and supportive. End with a kind closing message."
    )
    return {
        "final_plan": f"DEEP PRACTICE SESSION (3 Hours Study Plan)\n{'='*45}\n{response.content}",
        "messages": [f"[deep_prep] Generated deep session"]
    }


def route_after_decision(state: InterviewPrepState) -> str:
    if state.needs_deep_prep:
        return "deep"
    else:
        return "quick"


graph = StateGraph(InterviewPrepState)

graph.add_node("understand_interview_context", understand_interview_context)
graph.add_node("suggest_technical", technical_suggestion)
graph.add_node("suggest_behavioural", behavioural_suggestion)
graph.add_node("suggest_confidence", confidence_suggestion)
graph.add_node("pick_best_prep", pick_best_prep)
graph.add_node("quick_prep", quick_prep)
graph.add_node("deep_prep", deep_prep)

graph.add_edge(START, "understand_interview_context")

graph.add_edge("understand_interview_context", "suggest_technical")
graph.add_edge("understand_interview_context", "suggest_behavioural")
graph.add_edge("understand_interview_context", "suggest_confidence")

graph.add_edge("suggest_technical", "pick_best_prep")
graph.add_edge("suggest_behavioural", "pick_best_prep")
graph.add_edge("suggest_confidence", "pick_best_prep")

graph.add_conditional_edges(
    "pick_best_prep",
    route_after_decision,
    {
        "quick": "quick_prep",
        "deep": "deep_prep",
    }
)

graph.add_edge("quick_prep", END)
graph.add_edge("deep_prep", END)

app = graph.compile()


def run_interview_check(user_input: str):
    print("=" * 55)
    print("  INTERVIEW PREP SUGGESTER")
    print(f"  You said: \"{user_input}\"")
    print("=" * 55)

    result = app.invoke({
        "user_input": user_input,
        "messages": [],
    })

    print("\n" + "=" * 55)
    print("  YOUR PERSONALIZED PRACTICE")
    print("=" * 55)
    print(f"\n{result['final_plan']}")

    print("\n" + "-" * 55)
    print("  MESSAGE LOG")
    print("-" * 55)
    for msg in result["messages"]:
        print(f"  {msg}")

    return result


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  INTERVIEW PREP SUGGESTER")
    print("=" * 55)
    print("\n  Tell me about the interview you're preparing for and I'll suggest a")
    print("  personalized prep plan just for you.")
    print("  Type 'quit' to exit.\n")

    while True:
        user_input = input("  What interview are you preparing for? > ").strip()

        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  Take care of yourself. Goodbye!\n")
            break

        if not user_input:
            continue

        run_interview_check(user_input)
        print("\n")
