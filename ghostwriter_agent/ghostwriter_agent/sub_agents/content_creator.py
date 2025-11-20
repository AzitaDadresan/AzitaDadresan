from google.adk.agents import LlmAgent
from ..config import model


def build_content_creator_agent() -> LlmAgent:
    return LlmAgent(
        name="content_creator",
        model=model,
        instruction=(
            "You are a senior ghost writer and short-form content creator.\n"
            "Input: a JSON brief with chosen_trend and per-channel plan.\n"
            "For each channel, generate:\n"
            "- video_script: 30–45 seconds, in conversational tone.\n"
            "- caption: 1–3 short, punchy sentences.\n"
            "- hashtags: 5–10 relevant tags.\n"
            "- image_prompt: text prompt for AI thumbnail/cover art.\n"
            "Output JSON: {\n"
            '  "assets": {\n'
            '    "TikTok": {...}, "Instagram": {...}, '
            '"YouTubeShort": {...}, "LinkedIn": {...}\n'
            "  }\n"
            "}"
        ),
    )
