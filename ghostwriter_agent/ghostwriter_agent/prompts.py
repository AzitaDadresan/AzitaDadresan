"""Prompt templates extracted from `GhostWriter/main.ipynb`.

Keep these templates small and importable so agents use consistent wording.
"""

GROQ_SYSTEM_PROMPT = (
    "You are a Christian devotional writer. "
    "Do not use markdown. No asterisks, no bold formatting. "
    "Output clean paragraphs separated by line breaks."
)


GROQ_USER_PROMPT = (
    "Write a 160–200 word Christian devotional post for my blog at {site}. "
    "Begin with a scripture verse and reference. "
    "Encapsulate the verse inside <p>\" html tag. "
    "Then write a simple, warm reflection that connects the message of the verse to everyday life. "
    "End with one practical takeaway for the day. "
    "Keep paragraphs short and formatted cleanly for WordPress."
)


SAMPLE_RUN_PROMPT = (
    "Run one full GhostWriter cycle: detect trends, create content for "
    "TikTok, Instagram, YouTubeShort, and LinkedIn, publish (mock or real), "
    "and evaluate the performance. Brand topic: {topic}."
)


DAILY_TITLE_TEMPLATE = "Daily Serving: {date}"
