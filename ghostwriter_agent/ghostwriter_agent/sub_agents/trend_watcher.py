from google.adk.agents import LlmAgent
from ..config import model
from ..tools import fetch_trends_tool


def build_trend_watcher_agent() -> LlmAgent:
    return LlmAgent(
        name="trend_watcher",
        model=model,
        instruction=(
            "You are a trend analysis agent for a specific personal brand / ghost writer.\n"
            "- Always call the fetch_trends tool with the brand's niche, "
            "e.g. 'AI & career for women'.\n"
            "- Return the top 3 trends, ranked by velocity and relevance.\n"
            "- Output in JSON with fields: selected_trends (list), rationale (short text)."
        ),
        tools=[fetch_trends_tool],
    )
