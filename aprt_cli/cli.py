import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="APRT CLI")

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]):
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise typer.Exit(proc.returncode)


@app.command()
def preattack(
    base_dir: Path = typer.Option(Path.cwd(), help="Base directory"),
    last_epoch: int = typer.Option(0, help="Use epoch-<last_epoch> data"),
    target_backbone: str = typer.Option("llama3", help="llama2|llama3|vicuna"),
    attacker_checkpoint: Optional[str] = typer.Option(None, help="Attacker HF checkpoint path or None"),
    target_checkpoint: Optional[str] = typer.Option(None, help="Target HF checkpoint path or None"),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini if provider=api"),
    api_model_id: Optional[str] = typer.Option(None, help="Remote model id if provider=api"),
):
    cmd = [
        "bash", "primary_steps/pre_multi_attack.sh",
        str(base_dir), str(last_epoch), target_backbone,
    ]
    if attacker_checkpoint is not None:
        cmd.append(attacker_checkpoint)
    if target_checkpoint is not None:
        if attacker_checkpoint is None:
            cmd.append("")
        cmd.append(target_checkpoint)
    # provider args
    cmd.extend([provider])
    if api_provider:
        cmd.append(api_provider)
    if api_model_id:
        if not api_provider:
            cmd.append("")
        cmd.append(api_model_id)
    run(cmd)


@app.command()
def attack(
    base_dir: Path = typer.Option(Path.cwd(), help="Base directory"),
    epoch: int = typer.Option(1, help="Epoch id"),
    max_tokens: int = typer.Option(300),
    infer_freq: int = typer.Option(8),
    temperature: float = typer.Option(0.7),
    top_p: float = typer.Option(0.9),
    backbone: str = typer.Option("llama3"),
    attacker_checkpoint: Optional[str] = typer.Option(None),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini"),
    api_model_id: Optional[str] = typer.Option(None),
):
    cmd = [
        "bash", "primary_steps/multi_attack.sh",
        str(base_dir), str(epoch), str(max_tokens), str(infer_freq), str(temperature), str(top_p), backbone,
    ]
    if attacker_checkpoint is not None:
        cmd.append(attacker_checkpoint)
    cmd.append(provider)
    if api_provider:
        cmd.append(api_provider)
    if api_model_id:
        if not api_provider:
            cmd.append("")
        cmd.append(api_model_id)
    run(cmd)


@app.command()
def respond(
    base_dir: Path = typer.Option(Path.cwd()),
    epoch: int = typer.Option(1),
    max_tokens: int = typer.Option(600),
    infer_freq: int = typer.Option(8),
    temperature: float = typer.Option(0.7),
    top_p: float = typer.Option(0.9),
    backbone: str = typer.Option("llama3"),
    target_checkpoint: Optional[str] = typer.Option(None),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini"),
    api_model_id: Optional[str] = typer.Option(None),
):
    cmd = [
        "bash", "primary_steps/multi_attack_chat_response.sh",
        str(base_dir), str(epoch), str(max_tokens), str(infer_freq), str(temperature), str(top_p), backbone,
    ]
    if target_checkpoint is not None:
        cmd.append(target_checkpoint)
    cmd.append(provider)
    if api_provider:
        cmd.append(api_provider)
    if api_model_id:
        if not api_provider:
            cmd.append("")
        cmd.append(api_model_id)
    run(cmd)


@app.command()
def evaluate(
    base_dir: Path = typer.Option(Path.cwd()),
    now_epoch: int = typer.Option(1),
    attacker_max_tokens: int = typer.Option(300),
    responder_max_tokens: int = typer.Option(800),
    infer_freq: int = typer.Option(30),
    temperature: float = typer.Option(0.7),
    top_p: float = typer.Option(0.9),
    backbone: str = typer.Option("llama3"),
    attacker_checkpoint: Optional[str] = typer.Option(None),
    target_checkpoint: Optional[str] = typer.Option(None),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini"),
    api_model_id: Optional[str] = typer.Option(None),
):
    # attacker generation
    cmd1 = [
        "bash", "primary_steps/multi_evaluate.sh",
        str(base_dir), str(now_epoch), str(attacker_max_tokens), str(infer_freq), str(temperature), str(top_p), backbone,
    ]
    if attacker_checkpoint is not None:
        cmd1.append(attacker_checkpoint)
    cmd1.append(provider)
    if api_provider:
        cmd1.append(api_provider)
    if api_model_id:
        if not api_provider:
            cmd1.append("")
        cmd1.append(api_model_id)
    run(cmd1)

    # target responses
    cmd2 = [
        "bash", "primary_steps/multi_evaluate_chat_response.sh",
        str(base_dir), str(now_epoch), str(responder_max_tokens), str(infer_freq), str(temperature), str(top_p), backbone,
    ]
    if target_checkpoint is not None:
        cmd2.append(target_checkpoint)
    cmd2.append(provider)
    if api_provider:
        cmd2.append(api_provider)
    if api_model_id:
        if not api_provider:
            cmd2.append("")
        cmd2.append(api_model_id)
    run(cmd2)

    # guard + helpfulness + metrics
    run(["bash", "primary_steps/compute_safety_evaluate.sh", str(now_epoch), "target_stage1"])  # will manage transformer pin itself
    run(["bash", "primary_steps/compute_helpfulness_evaluate.sh", str(now_epoch), str(base_dir), "target_stage1"]) 


if __name__ == "__main__":
    app()