import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path

# Mock external modules that might perform networking or heavy setup
with patch("browser_use.ChatGoogle"), \
     patch("browser_use.BrowserSession"), \
     patch("browser_use.Agent"), \
     patch("optexity.inference.core.interaction.handle_agentic_task.convert_history_to_cached_actions"):

    from optexity.schema.memory import Memory
    from optexity.schema.task import Task
    from optexity.schema.actions.interaction_action import AgenticTask
    from optexity.inference.core.interaction.handle_agentic_task import handle_agentic_task
    from optexity.inference.infra.browser import Browser

# Setup test case
async def test_token_usage_propagation():
    # 1. Create a mock Memory
    memory = Memory()
    memory.token_usage.input_tokens = 0
    memory.token_usage.output_tokens = 0
    memory.token_usage.total_tokens = 0
    memory.token_usage.input_cost = 0.0
    memory.token_usage.output_cost = 0.0
    memory.token_usage.total_cost = 0.0

    # 2. Create a mock Task
    task = MagicMock(spec=Task)
    task.automation = MagicMock()
    task.automation.url = "https://example.com"
    task.logs_directory = Path("/tmp/mock_logs")
    task.input_parameters = {}

    # 3. Create a mock Browser
    browser = MagicMock(spec=Browser)
    browser.cdp_url = "ws://mock-cdp"

    # 4. Create action
    action = MagicMock(spec=AgenticTask)
    action.task = "mock task"
    action.backend = "browser_use"
    action.keep_alive = True
    action.max_steps = 10
    action.use_vision = True

    # 5. Mock AgentHistoryList and UsageSummary
    from browser_use.agent.views import AgentHistoryList
    from browser_use.tokens.views import UsageSummary

    mock_usage = UsageSummary(
        total_prompt_tokens=1500,
        total_prompt_cost=0.03,
        total_prompt_cached_tokens=200,
        total_prompt_cached_cost=0.005,
        total_completion_tokens=500,
        total_completion_cost=0.01,
        total_tokens=2200,
        total_cost=0.045,
        entry_count=1,
    )

    mock_history = MagicMock(spec=AgentHistoryList)
    mock_history.usage = mock_usage
    mock_history.is_done.return_value = True

    # 6. Patch and run handle_agentic_task
    with patch("optexity.inference.core.interaction.handle_agentic_task.ChatGoogle") as mock_chat, \
         patch("optexity.inference.core.interaction.handle_agentic_task.BrowserSession") as mock_session, \
         patch("optexity.inference.core.interaction.handle_agentic_task.Agent") as mock_agent_cls, \
         patch("optexity.inference.core.interaction.handle_agentic_task.convert_history_to_cached_actions") as mock_convert:

        # Configure mocks
        mock_agent = MagicMock()
        mock_agent.run = AsyncMock(return_value=mock_history)
        mock_agent.stop = MagicMock()
        mock_agent.browser_session = AsyncMock()
        mock_agent_cls.return_value = mock_agent

        mock_convert.return_value = []

        # Run the action handler
        await handle_agentic_task(action, task, memory, browser)

        # Assert token usage is correctly added to memory
        print(f"Memory input tokens: {memory.token_usage.input_tokens}")
        print(f"Memory output tokens: {memory.token_usage.output_tokens}")
        print(f"Memory total tokens: {memory.token_usage.total_tokens}")
        print(f"Memory input cost: {memory.token_usage.input_cost}")
        print(f"Memory output cost: {memory.token_usage.output_cost}")
        print(f"Memory total cost: {memory.token_usage.total_cost}")

        assert memory.token_usage.input_tokens == 1500
        assert memory.token_usage.output_tokens == 500
        assert memory.token_usage.total_tokens == 2200
        assert abs(memory.token_usage.input_cost - 0.035) < 1e-6
        assert abs(memory.token_usage.output_cost - 0.01) < 1e-6
        assert abs(memory.token_usage.total_cost - 0.045) < 1e-6
        print("✅ Token usage propagation test passed!")

if __name__ == "__main__":
    asyncio.run(test_token_usage_propagation())
