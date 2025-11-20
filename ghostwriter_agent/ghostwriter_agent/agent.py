import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import InMemoryRunner

from .config import model  # <- uses Gemini from config.py
from .prompts import SAMPLE_RUN_PROMPT
from .sub_agents import (
    build_trend_watcher_agent,
    build_content_strategist_agent,
    build_content_creator_agent,
    build_publisher_agent,
    build_evaluator_agent,
)

# Build sub-agents
trend_watcher = build_trend_watcher_agent()
content_strategist = build_content_strategist_agent()
content_creator = build_content_creator_agent()
publisher_agent = build_publisher_agent()
evaluator_agent = build_evaluator_agent()

# Main orchestrator: now LlmAgent (not plain Agent)
interactive_ghostwriter_agent = LlmAgent(
    name="interactive_ghostwriter_agent",
    model=model,  # 🔹 this gives it a canonical_model
    description=(
        "A multi-agent ghost writer / brand visibility system that:\n"
        "- Watches trends\n"
        "- Plans cross-channel content\n"
        "- Writes scripts & captions\n"
        "- (Mock) publishes\n"
        "- Evaluates performance and suggests improvements."
    ),
    instruction=(
        "You are the orchestrator for a ghost writer agent team.\n"
        "Given a user request about a personal/brand topic, you:\n"
        "1. Delegate to the trend_watcher to identify hot topics.\n"
        "2. Delegate to content_strategist to design a 4-channel plan.\n"
        "3. Delegate to content_creator to generate scripts, captions, and prompts.\n"
        "4. Delegate to publisher_agent to schedule posts (mock).\n"
        "5. Delegate to evaluator_agent to analyze performance and suggest next steps.\n"
        "Return a concise final summary of the campaign and key learnings."
    ),
    sub_agents=[
        trend_watcher,
        content_strategist,
        content_creator,
        publisher_agent,
        evaluator_agent,
    ],
)

# Runner for local CLI / tests
runner = InMemoryRunner(agent=interactive_ghostwriter_agent)


async def run_demo() -> None:
    user_prompt = (
        SAMPLE_RUN_PROMPT.format(topic="AI, career, and women in tech")
    )

    # Use run_debug (works across the Kaggle / course version of ADK)
    result = await runner.run_debug(user_prompt)

    print("\n=== RAW DEBUG RESULT FROM ADK ===\n")
    print(result)



if __name__ == "__main__":
    asyncio.run(run_demo())
