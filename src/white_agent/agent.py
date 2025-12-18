"""White agent implementation - returns computer_13 actions using GPT-4o."""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import dotenv
import openai
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, AgentCapabilities, AgentSkill
from a2a.utils import new_agent_text_message

from src.my_util import extract_json


dotenv.load_dotenv()


ACTION_SCHEMA_TEXT = """
You are a vision-capable desktop agent. Return exactly one JSON action wrapped in <json>...</json>.
Allowed action_type values (computer_13):
- MOVE_TO: x, y
- CLICK: button (left/right/middle, optional), x, y (optional), num_clicks (optional)
- MOUSE_DOWN/MOUSE_UP: button (left/right/middle, optional)
- RIGHT_CLICK/DOUBLE_CLICK: x, y (optional)
- DRAG_TO: x, y
- SCROLL: dx, dy
- TYPING: text
- PRESS/KEY_DOWN/KEY_UP: key
- HOTKEY: keys (list of key strings)
- WAIT/FAIL/DONE: action_type only
Screen origin is top-left.
Do not include any text outside the <json>...</json> wrapper.
""".strip()


class WhiteAgentExecutor(AgentExecutor):
    def __init__(self) -> None:
        self.ctx_messages: Dict[str, list] = {}
        self.client = openai.OpenAI()

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        ctx_id = context.context_id or "default"
        if ctx_id not in self.ctx_messages:
            self.ctx_messages[ctx_id] = [
                {"role": "system", "content": ACTION_SCHEMA_TEXT}
            ]

        user_input = context.get_user_input()
        self.ctx_messages[ctx_id].append({"role": "user", "content": user_input})

        response = self.client.chat.completions.create(
            model=os.getenv("WHITE_AGENT_MODEL", "gpt-4o"),
            messages=self.ctx_messages[ctx_id],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        self.ctx_messages[ctx_id].append({"role": "assistant", "content": content})

        # Echo the raw content back; green agent will parse the JSON action.
        await event_queue.enqueue_event(
            new_agent_text_message(content, context_id=context.context_id)
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def prepare_white_agent_card(url: str) -> AgentCard:
    skill = AgentSkill(
        id="computer_13_action",
        name="Computer 13 Action Selector",
        description="Selects the next computer_13 action given prompt/observation.",
        tags=["white agent", "osworld"],
        examples=[],
    )
    return AgentCard(
        protocolVersion="0.3.0",
        name="osworld_white_agent",
        description="A general-purpose white agent producing computer_13 actions.",
        url=url,
        version="0.1.0",
        default_input_modes=["text/plain", "image/png"],
        default_output_modes=["text/plain"],
        capabilities=AgentCapabilities(),
        skills=[skill],
    )


def start_white_agent(agent_name: str = "osworld_white_agent", host: str = "localhost", port: int = 9002) -> None:
    print("Starting white agent...")
    url = f"http://{host}:{port}"
    url = os.getenv("AGENT_URL", url)
    card = prepare_white_agent_card(url)

    request_handler = DefaultRequestHandler(
        agent_executor=WhiteAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=card,
        http_handler=request_handler,
    )

    import uvicorn

    uvicorn.run(app.build(), host=host, port=port)
