from google.adk.agents import LlmAgent
from ..config import model


def build_image_generator_agent() -> LlmAgent:
    return LlmAgent(
        name="image_generator",
        model=model,
        instruction=(
            "You are an image generation agent for the ghost writer system.\n"
            "You receive image prompts and descriptions from content creators.\n"
            "Tasks:\n"
            "1. Refine and optimize image prompts for clarity and visual appeal.\n"
            "2. Suggest appropriate styles (e.g., 'professional', 'casual', 'minimalist', 'vibrant').\n"
            "3. Generate image generation requests with optimized prompts.\n"
            "Output JSON: {\n"
            '  "refined_prompt": "...", \n'
            '  "style": "...", \n'
            '  "suggestions": [...]\n'
            "}"
        ),
    )

