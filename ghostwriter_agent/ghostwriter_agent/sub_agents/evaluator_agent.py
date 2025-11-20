from google.adk.agents import LlmAgent
from ..config import model
from ..tools import analytics_tool


def build_evaluator_agent() -> LlmAgent:
    return LlmAgent(
        name="evaluator_agent",
        model=model,
        instruction=(
            "You are a performance analyst for ghost-written social campaigns.\n"
            "You must ALWAYS call get_mock_analytics to fetch performance metrics.\n"
            "Then:\n"
            "1. Compute which channel performed best (views, engagement).\n"
            "2. Suggest 3 concrete improvements for the next campaign.\n"
            "3. Output JSON: {\n"
            '   "best_channel": ..., \n'
            '   "summary": ..., \n'
            '   "suggestions": [...]\n'
            "}"
        ),
        tools=[analytics_tool],
    )
