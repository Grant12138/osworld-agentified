"""OSWorld runner service for Python 3.10 environments."""

from __future__ import annotations

import base64
import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
    SendMessageResponse,
    SendMessageSuccessResponse,
    Task,
    TextPart,
    DataPart,
    FilePart,
)
from a2a.utils import get_text_parts

from desktop_env.desktop_env import DesktopEnv


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
    sleep_after_execution: float = 3.0
    max_steps: int = 15
    reset_wait_seconds: int = 60
    post_eval_wait_seconds: int = 20
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


class EvaluateRequest(BaseModel):
    white_agent_url: str
    osworld_config: Dict[str, Any] = {}


app = FastAPI()


ACTION_TYPES = [
    "MOVE_TO",
    "CLICK",
    "MOUSE_DOWN",
    "MOUSE_UP",
    "RIGHT_CLICK",
    "DOUBLE_CLICK",
    "DRAG_TO",
    "SCROLL",
    "TYPING",
    "PRESS",
    "KEY_DOWN",
    "KEY_UP",
    "HOTKEY",
    "WAIT",
    "FAIL",
    "DONE",
]


def parse_tags(str_with_tags: str) -> Dict[str, str]:
    tags = re.findall(r"<(.*?)>(.*?)</\1>", str_with_tags, re.DOTALL)
    return {tag: content.strip() for tag, content in tags}


def extract_json(text: str) -> Optional[dict]:
    tags = parse_tags(text)
    if "json" in tags:
        try:
            return json.loads(tags["json"])
        except json.JSONDecodeError:
            return None
    fence_match = re.search(r"```(?:json)?\n(.*?)\n```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            return None
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            return None
    return None


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


def _load_task_meta(config: OSWorldConfig) -> Dict[str, List[str]]:
    if config.task_ids:
        return config.task_ids
    if not config.test_all_meta_path:
        raise ValueError("test_all_meta_path is required when task_ids is not provided.")
    with open(config.test_all_meta_path, "r", encoding="utf-8") as f:
        test_all_meta = json.load(f)
    if config.domain != "all":
        if config.domain not in test_all_meta:
            raise ValueError(f"Domain not found in meta file: {config.domain}")
        test_all_meta = {config.domain: test_all_meta[config.domain]}
    return test_all_meta


def _action_schema_text(screen_width: int, screen_height: int) -> str:
    return (
        "Return exactly one JSON object wrapped in <json>...</json>. "
        "The object must include action_type and any required parameters.\n"
        f"Screen size: {screen_width}x{screen_height} (origin is top-left).\n"
        "Allowed action_type values:\n"
        f"- {', '.join(ACTION_TYPES)}\n"
        "Parameters by action:\n"
        "- MOVE_TO: x, y (float/int)\n"
        "- CLICK: button (left/right/middle, optional), x, y (optional), num_clicks (optional)\n"
        "- MOUSE_DOWN/MOUSE_UP: button (left/right/middle, optional)\n"
        "- RIGHT_CLICK/DOUBLE_CLICK: x, y (optional)\n"
        "- DRAG_TO: x, y\n"
        "- SCROLL: dx, dy (ints; positive/negative)\n"
        "- TYPING: text\n"
        "- PRESS/KEY_DOWN/KEY_UP: key\n"
        "- HOTKEY: keys (list of key strings)\n"
        "- WAIT/FAIL/DONE: action_type only\n"
        "Do not include extra text outside the JSON wrapper."
    )


def _build_prompt(
    instruction: str,
    screen_width: int,
    screen_height: int,
    step_idx: int,
    max_steps: int,
    last_action: Optional[Any],
) -> str:
    last_action_str = json.dumps(last_action) if last_action is not None else "none"
    return (
        f"Task instruction:\n{instruction}\n\n"
        f"Step {step_idx + 1} of {max_steps}\n"
        f"Previous action: {last_action_str}\n\n"
        "You will receive a screenshot as a file attachment. "
        "Decide the next action using the schema below.\n\n"
        f"{_action_schema_text(screen_width, screen_height)}"
    )


def _encode_screenshot(screenshot: Any) -> Optional[str]:
    if screenshot is None:
        return None
    if isinstance(screenshot, bytes):
        return base64.b64encode(screenshot).decode("utf-8")
    if isinstance(screenshot, str):
        return screenshot
    return None


def _normalize_action(action: Any) -> Any:
    if isinstance(action, str):
        return action
    if not isinstance(action, dict):
        raise ValueError("Action must be a dict or string.")
    if "action_type" in action and isinstance(action["action_type"], str):
        action["action_type"] = action["action_type"].upper()
    return action


def _message_from_task(task: Task) -> Optional[Message]:
    if task.status and task.status.message:
        return task.status.message
    if task.history:
        for msg in reversed(task.history):
            if msg.role == "agent":
                return msg
    return None


def _extract_action_from_message(message: Message) -> Any:
    for part in message.parts:
        if isinstance(part, DataPart) or getattr(part, "kind", None) == "data":
            return getattr(part, "data", None) or part.get("data")

    texts = get_text_parts(message.parts)
    if not texts:
        for part in message.parts:
            if isinstance(part, TextPart) or getattr(part, "kind", None) == "text":
                text_value = getattr(part, "text", None) or part.get("text")
                if text_value:
                    texts.append(text_value)

    for text in texts:
        normalized = text.strip()
        if normalized in {"WAIT", "FAIL", "DONE"}:
            return {"action_type": normalized}
        action_obj = extract_json(normalized)
        if action_obj is not None:
            return action_obj

    raise ValueError("No actionable response found in white agent message.")


async def _get_agent_card(url: str):
    httpx_client = httpx.AsyncClient()
    resolver = A2ACardResolver(httpx_client=httpx_client, base_url=url)
    return await resolver.get_agent_card()


async def _send_message(
    url: str,
    parts: List[Part],
    task_id: Optional[str] = None,
    context_id: Optional[str] = None,
) -> SendMessageResponse:
    card = await _get_agent_card(url)
    httpx_client = httpx.AsyncClient(timeout=120.0)
    client = A2AClient(httpx_client=httpx_client, agent_card=card)

    message_id = os.urandom(16).hex()
    params = MessageSendParams(
        message=Message(
            role=Role.user,
            parts=parts,
            message_id=message_id,
            task_id=task_id,
            context_id=context_id,
        )
    )
    request_id = os.urandom(16).hex()
    req = SendMessageRequest(id=request_id, params=params)
    return await client.send_message(request=req)


async def _ask_white_agent(
    white_agent_url: str,
    instruction: str,
    screenshot_bytes: Optional[bytes],
    screen_width: int,
    screen_height: int,
    step_idx: int,
    max_steps: int,
    last_action: Optional[Any],
    context_id: Optional[str],
    task_id: Optional[str],
) -> Tuple[Any, Optional[str], Optional[str]]:
    prompt = _build_prompt(
        instruction=instruction,
        screen_width=screen_width,
        screen_height=screen_height,
        step_idx=step_idx,
        max_steps=max_steps,
        last_action=last_action,
    )

    parts: List[Part] = [TextPart(text=prompt)]
    screenshot_b64 = _encode_screenshot(screenshot_bytes)
    if screenshot_b64:
        parts.append(
            FilePart(
                file={
                    "name": "screenshot.png",
                    "mimeType": "image/png",
                    "bytes": screenshot_b64,
                }
            )
        )

    response = await _send_message(
        white_agent_url, parts, task_id=task_id, context_id=context_id
    )
    res_root = response.root
    if not isinstance(res_root, SendMessageSuccessResponse):
        raise ValueError("Unexpected A2A response type from white agent.")
    res_result = res_root.result

    next_context_id = context_id
    next_task_id = task_id
    if isinstance(res_result, Message):
        next_context_id = res_result.context_id or context_id
        next_task_id = res_result.task_id or task_id
        action = _extract_action_from_message(res_result)
    elif isinstance(res_result, Task):
        next_context_id = res_result.context_id or context_id
        next_task_id = res_result.id or task_id
        message = _message_from_task(res_result)
        if message is None:
            raise ValueError("Task response did not include a usable message.")
        action = _extract_action_from_message(message)
    else:
        raise ValueError("Unexpected result type in A2A response.")

    return _normalize_action(action), next_context_id, next_task_id


def _build_env(config: OSWorldConfig) -> DesktopEnv:
    if not config.path_to_vm:
        raise ValueError("path_to_vm is required to start DesktopEnv.")
    return DesktopEnv(
        provider_name=config.provider_name,
        path_to_vm=config.path_to_vm,
        os_type=config.os_type,
        action_space=config.action_space,
        screen_size=(config.screen_width, config.screen_height),
        headless=config.headless,
        require_a11y_tree=config.require_a11y_tree,
        require_terminal=config.require_terminal,
        enable_proxy=config.enable_proxy,
        client_password=config.client_password,
        snapshot_name=config.snapshot_name,
    )


async def run_osworld_assessment(
    white_agent_url: str, config: OSWorldConfig
) -> Dict[str, Any]:
    config = _resolve_paths(config)
    test_all_meta = _load_task_meta(config)
    env = _build_env(config)
    results: Dict[str, Any] = {
        "tasks": [],
        "successes": 0,
        "total": 0,
    }

    try:
        for domain, task_ids in test_all_meta.items():
            for task_id in task_ids:
                example_path = os.path.join(
                    config.test_config_base_dir or "",
                    "examples",
                    domain,
                    f"{task_id}.json",
                )
                with open(example_path, "r", encoding="utf-8") as f:
                    example = json.load(f)

                instruction = example.get("instruction", "")
                env.reset(task_config=example)
                time.sleep(config.reset_wait_seconds)
                obs = env._get_obs()

                done = False
                step_idx = 0
                last_action = None
                context_id = None
                task_context_id = None

                while not done and step_idx < config.max_steps:
                    action, context_id, task_context_id = await _ask_white_agent(
                        white_agent_url=white_agent_url,
                        instruction=instruction,
                        screenshot_bytes=obs.get("screenshot"),
                        screen_width=config.screen_width,
                        screen_height=config.screen_height,
                        step_idx=step_idx,
                        max_steps=config.max_steps,
                        last_action=last_action,
                        context_id=context_id,
                        task_id=task_context_id,
                    )
                    last_action = action
                    obs, reward, done, info = env.step(
                        action, pause=config.sleep_after_execution
                    )
                    step_idx += 1

                time.sleep(config.post_eval_wait_seconds)
                score = env.evaluate()
                success = score >= 1.0
                results["tasks"].append(
                    {
                        "domain": domain,
                        "task_id": task_id,
                        "score": score,
                        "success": success,
                    }
                )
                results["total"] += 1
                results["successes"] += 1 if success else 0
    finally:
        env.close()

    results["success_rate"] = (
        results["successes"] / results["total"] if results["total"] else 0.0
    )
    return results


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest) -> Dict[str, Any]:
    config = OSWorldConfig(**req.osworld_config)
    if not config.path_to_vm and not os.getenv("OSWORLD_VM_PATH"):
        raise HTTPException(status_code=400, detail="path_to_vm is required")
    try:
        return await run_osworld_assessment(req.white_agent_url, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("OSWORLD_RUNNER_HOST", "127.0.0.1")
    port = int(os.getenv("OSWORLD_RUNNER_PORT", "9010"))
    uvicorn.run(app, host=host, port=port)
