"""Green agent implementation - manages OSWorld assessment and evaluation."""

from __future__ import annotations

import json
import os
import tomllib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import dotenv
import httpx
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message

from src.my_util import parse_tags


dotenv.load_dotenv()


@dataclass
class OSWorldConfig:
    provider_name: str = "vmware"
    path_to_vm: Optional[str] = None
    os_type: str = "Ubuntu"
    action_space: str = "computer_13"
    headless: bool = False
    screen_width: int = 1920
    screen_height: int = 1080
    sleep_after_execution: float = 1.0
    max_steps: int = 10
    reset_wait_seconds: int = 5
    post_eval_wait_seconds: int = 5
    require_a11y_tree: bool = False
    require_terminal: bool = False
    enable_proxy: bool = False
    client_password: str = "password"
    snapshot_name: str = "init_state"
    osworld_base_dir: Optional[str] = None
    test_config_base_dir: Optional[str] = None
    test_all_meta_path: Optional[str] = None
    domain: str = "all"
    task_ids: Optional[Dict[str, List[str]]] = None


def load_agent_card_toml(agent_name: str) -> dict:
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


def _resolve_paths(config: OSWorldConfig) -> OSWorldConfig:
    if not config.path_to_vm:
        config.path_to_vm = os.getenv("OSWORLD_VM_PATH")
    base_dir = config.osworld_base_dir or os.getenv("OSWORLD_BASE_DIR")
    if config.test_config_base_dir is None and base_dir:
        config.test_config_base_dir = os.path.join(base_dir, "evaluation_examples")
    if config.test_all_meta_path is None:
        env_path = os.getenv("OSWORLD_TEST_META_PATH")
        if env_path:
            config.test_all_meta_path = env_path
        elif base_dir:
            config.test_all_meta_path = os.path.join(
                base_dir, "evaluation_examples", "test_small.json"
            )
    return config


def _runner_url(config: OSWorldConfig) -> str:
    return os.getenv("OSWORLD_RUNNER_URL", "http://localhost:9010")


async def run_osworld_assessment_remote(
    white_agent_url: str, config: OSWorldConfig
) -> Dict[str, Any]:
    config = _resolve_paths(config)
    payload = {
        "white_agent_url": white_agent_url,
        "osworld_config": config.__dict__,
    }
    runner_url = _runner_url(config).rstrip("/")
    async with httpx.AsyncClient(timeout=None) as client:
        response = await client.post(f"{runner_url}/evaluate", json=payload)
        response.raise_for_status()
        return response.json()


class OSWorldGreenAgentExecutor(AgentExecutor):
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags.get("white_agent_url") or tags.get("white_agent")
        if not white_agent_url:
            raise ValueError("white_agent_url is required in the task payload.")

        config_dict: Dict[str, Any] = {}
        if "osworld_config" in tags:
            config_dict = json.loads(tags["osworld_config"])
        elif "env_config" in tags:
            config_dict = json.loads(tags["env_config"])

        config = OSWorldConfig(**config_dict)

        # Run the assessment synchronously so AgentBeats waits for real results.
        results = await run_osworld_assessment_remote(white_agent_url, config)
        summary = {
            "success_rate": results.get("success_rate"),
            "successes": results.get("successes"),
            "total": results.get("total"),
        }
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished OSWorld evaluation. Metrics: {json.dumps(summary)}\n"
            )
        )

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


def start_green_agent(
    agent_name: str = "osworld_green_agent", host: str = "localhost", port: int = 9001
) -> None:
    print("Starting OSWorld green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = os.getenv("AGENT_URL", url)

    request_handler = DefaultRequestHandler(
        agent_executor=OSWorldGreenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    uvicorn.run(app.build(), host=host, port=port)
