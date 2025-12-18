# Agentify Example: OSWorld

Example code for agentifying OSWorld using A2A and MCP standards.

## Project Structure

```
src/
├── green_agent/    # Assessment manager agent
├── white_agent/    
└── my_util/        # A2A helpers and tag parsing
osworld_runner/     # Python 3.10 OSWorld runner service
```

## Installation

```bash
uv sync
```

Or with pip:

```bash
pip install -e .
```

## OSWorld Runner (Python 3.10)

The green agent runs on Python 3.13 (AgentBeats), but OSWorld requires Python 3.10/3.11.
Run the OSWorld runner in a separate Python 3.10 env:

```bash
cd /Users/grantluo/Desktop/agentos-green/agentify-example-osworld
python3.10 -m venv .venv-osworld
source .venv-osworld/bin/activate
pip install -r osworld_runner/requirements.txt

export OSWORLD_BASE_DIR="/Users/you/OSWorld"
export OSWORLD_VM_PATH="/Users/you/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx"

python osworld_runner/runner.py
```

The runner listens on `http://localhost:9010` by default.

## Usage

Set OSWorld paths (either via env vars or in the task payload):

```bash
export OSWORLD_BASE_DIR="/Users/you/OSWorld"
export OSWORLD_VM_PATH="/Users/you/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx"
export OSWORLD_RUNNER_URL="http://localhost:9010"
```

Start the green agent locally (Python 3.13 env):

```bash
python main.py green
```

Start the white agent locally:

```bash
python main.py white
```

Or start via AgentBeats controller (expects `HOST` and `AGENT_PORT`):

```bash
python main.py run
```

## Task Payload Format

Send a task to the green agent with tag-based config (same pattern as the tau example):

```
<white_agent_url>
http://localhost:9002/
</white_agent_url>
<osworld_config>
{
  "provider_name": "vmware",
  "path_to_vm": "/Users/you/OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx",
  "action_space": "computer_13",
  "screen_width": 1920,
  "screen_height": 880,
  "sleep_after_execution": 3,
  "max_steps": 15,
  "osworld_base_dir": "/Users/you/OSWorld",
  "test_all_meta_path": "/Users/you/OSWorld/evaluation_examples/test_small.json"
}
</osworld_config>
```

### Supported config fields

- `provider_name` (default: `vmware`)
- `path_to_vm` (required if not set via `OSWORLD_VM_PATH`)
- `os_type` (default: `Ubuntu`)
- `action_space` (default: `computer_13`)
- `headless` (default: `false`)
- `screen_width`, `screen_height`
- `sleep_after_execution` (seconds)
- `max_steps`
- `reset_wait_seconds` (default: 60)
- `post_eval_wait_seconds` (default: 20)
- `osworld_base_dir`
- `test_config_base_dir` (defaults to `<osworld_base_dir>/evaluation_examples`)
- `test_all_meta_path` (defaults to `<osworld_base_dir>/evaluation_examples/test_small.json`)
- `domain` (default: `all`)
- `task_ids` (optional dict mapping domain -> list of task IDs)

## White Agent Response Format

The green agent sends a screenshot (A2A FilePart) plus a short text prompt. The white agent must respond with one JSON action wrapped in `<json>...</json>` tags, for example:

```
<json>{"action_type":"CLICK","x":120,"y":400}</json>
```

Allowed `action_type` values are defined by OSWorld `computer_13`:
MOVE_TO, CLICK, MOUSE_DOWN, MOUSE_UP, RIGHT_CLICK, DOUBLE_CLICK, DRAG_TO,
SCROLL, TYPING, PRESS, KEY_DOWN, KEY_UP, HOTKEY, WAIT, FAIL, DONE.
