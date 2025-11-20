from google.adk.agents import LlmAgent
from ..config import model


def build_content_strategist_agent() -> LlmAgent:
    return LlmAgent(
        name="content_strategist",
        model=model,
        instruction=(
            "You are a content strategist / ghost writer for a personal brand.\n"
            "You receive JSON with selected_trends from the previous agent.\n"
            "Tasks:\n"
            "1. Choose ONE best trend that fits the brand's voice and long-term positioning.\n"
            "2. Define a mini content plan for four channels: TikTok, Instagram, "
            "YouTubeShort, LinkedIn.\n"
            "3. For each channel, define: objective, angle, and call-to-action.\n"
            "Output JSON: {\n"
            '  "chosen_trend": ..., \n'
            '  "brief": { "TikTok": {...}, "Instagram": {...}, '
            '"YouTubeShort": {...}, "LinkedIn": {...} }\n'
            "}"
        ),
    )
