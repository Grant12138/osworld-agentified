"""CLI entry point for agentify-example-osworld."""

import typer
from pydantic_settings import BaseSettings

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent


class OSWorldSettings(BaseSettings):
    role: str = "green"
    host: str = "127.0.0.1"
    agent_port: int = 9001


app = typer.Typer(help="Agentified OSWorld - Standardized agent assessment framework")


@app.command()
def green() -> None:
    """Start the green agent (assessment manager)."""
    start_green_agent()


@app.command()
def white() -> None:
    """Start the white agent (target)."""
    start_white_agent()


@app.command()
def run() -> None:
    """Start the green agent with AgentBeats controller settings."""
    settings = OSWorldSettings()
    if settings.role == "green":
        start_green_agent(host=settings.host, port=settings.agent_port)
    elif settings.role == "white":
        start_white_agent(host=settings.host, port=settings.agent_port)
    else:
        raise ValueError(f"Unsupported role: {settings.role}")


if __name__ == "__main__":
    app()
