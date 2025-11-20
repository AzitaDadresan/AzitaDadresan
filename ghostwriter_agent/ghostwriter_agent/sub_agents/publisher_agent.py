from google.adk.agents import LlmAgent
from ..config import model
from ..tools import publish_tool


def build_publisher_agent() -> LlmAgent:
    return LlmAgent(
        name="publisher_agent",
        model=model,
        instruction=(
            "You are a publishing coordinator agent for the ghost writer system.\n"
            "You receive generated assets for all channels.\n"
            "1. Transform them into a payload for the publish_or_schedule tool: {\n"
            '   "items": [\n'
            '      {"channel": ..., "caption": ..., "script": ..., '
            '"scheduled_time": ...}, ...\n'
            "   ]\n"
            "}\n"
            "2. ALWAYS call publish_or_schedule.\n"
            "3. Return a human-readable summary of what was scheduled where, "
            "plus the raw tool output.\n"
        ),
        tools=[publish_tool],
    )
