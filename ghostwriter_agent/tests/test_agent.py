import asyncio

from ghostwriter_agent.agent import runner


def test_interactive_ghostwriter_agent_runs():
    async def _run():
        resp = await runner.run_debug(
            "Run one short ghost-writer cycle for the topic "
            "'AI, career, and women in tech'."
        )
        assert resp.output_text  # at least something came back

    asyncio.run(_run())
