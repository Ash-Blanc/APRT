import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="APRT CLI - Automated Progressive Red Teaming")

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(cmd: list[str]):
    proc = subprocess.run(cmd, cwd=str(ROOT))
    if proc.returncode != 0:
        raise typer.Exit(proc.returncode)


def load_default_models(config_path: Path = ROOT / "load_init_model.json") -> dict:
    if not config_path.exists():
        return {}
    try:
        return json.loads(config_path.read_text())
    except Exception:
        return {}


@app.command()
def init():
    """Initialize experiment directories and seed data."""
    run(["bash", "init_exp.sh"])


@app.command()
def preattack(
    base_dir: Path = typer.Option(Path.cwd(), help="Base directory"),
    last_epoch: int = typer.Option(0, help="Use epoch-<last_epoch> data"),
    target_backbone: str = typer.Option("auto", help="llama2|llama3|vicuna|auto"),
    attacker_checkpoint: Optional[str] = typer.Option(None, help="Local attacker checkpoint; omit for API"),
    target_checkpoint: Optional[str] = typer.Option(None, help="Local target checkpoint; omit for API"),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini if provider=api"),
    api_model_id: Optional[str] = typer.Option(None, help="Remote model id if provider=api"),
):
    # Default backbone
    if target_backbone == "auto":
        target_backbone = "llama3"
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
    backbone: str = typer.Option("auto", help="llama2|llama3|vicuna|auto"),
    attacker_checkpoint: Optional[str] = typer.Option(None, help="Local attacker checkpoint; omit for API"),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini"),
    api_model_id: Optional[str] = typer.Option(None),
):
    if backbone == "auto":
        backbone = "llama3"
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
    backbone: str = typer.Option("auto", help="llama2|llama3|vicuna|auto"),
    target_checkpoint: Optional[str] = typer.Option(None, help="Local target checkpoint; omit for API"),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini"),
    api_model_id: Optional[str] = typer.Option(None),
):
    if backbone == "auto":
        backbone = "llama3"
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
def score(
    epoch: int = typer.Option(1, help="Epoch id to score"),
    target_stage: str = typer.Option("pre_attack_target_stage1", help="target_stage1|pre_attack_target_stage1"),
    base_dir: Path = typer.Option(Path.cwd(), help="Base directory for helpfulness"),
    model_role: str = typer.Option("target_pre-attack", help="Role tag for helpfulness scorer"),
):
    # safety (Llama-Guard)
    run(["bash", "primary_steps/compute_safety_guard.sh", str(epoch), target_stage])
    # helpfulness (UltraRM)
    run(["bash", "primary_steps/compute_helpfulness.sh", str(epoch), str(base_dir), model_role])


@app.command()
def select(
    now_epoch: int = typer.Option(1, help="Current epoch"),
):
    """Select data for next round based on ASR and helpfulness."""
    record_dir = ROOT / f"record/epoch-{now_epoch-1}"
    merged = record_dir / "attack.json.attack_output_response.guard.helpful"
    run(["bash", "-lc", f"cat {record_dir}/attack.json.*.attack_output_response.guard.helpful > {merged}"])
    run(["python3", "scripts/sort_incremental_red_by_asr.py", str(merged), str(merged)+".asr_sort", "False"])
    run(["python3", "scripts/active_learning_selection_red.py", str(merged)+".asr_sort", str(merged)+".asr_sort.red_next", "0.2", "0.58", "100"])


@app.command()
def evaluate(
    base_dir: Path = typer.Option(Path.cwd()),
    now_epoch: int = typer.Option(1),
    attacker_max_tokens: int = typer.Option(300),
    responder_max_tokens: int = typer.Option(800),
    infer_freq: int = typer.Option(30),
    temperature: float = typer.Option(0.7),
    top_p: float = typer.Option(0.9),
    backbone: str = typer.Option("auto", help="llama2|llama3|vicuna|auto"),
    attacker_checkpoint: Optional[str] = typer.Option(None),
    target_checkpoint: Optional[str] = typer.Option(None),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini"),
    api_model_id: Optional[str] = typer.Option(None),
):
    if backbone == "auto":
        backbone = "llama3"
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


@app.command()
def pipeline(
    base_dir: Path = typer.Option(Path.cwd()),
    now_epoch: int = typer.Option(1),
    last_epoch: Optional[int] = typer.Option(None, help="Defaults to now_epoch-1"),
    backbone: str = typer.Option("auto", help="llama2|llama3|vicuna|auto"),
    attacker_checkpoint: Optional[str] = typer.Option(None),
    target_checkpoint: Optional[str] = typer.Option(None),
    provider: str = typer.Option("local", help="local|api"),
    api_provider: Optional[str] = typer.Option(None, help="hf|gemini"),
    api_model_id: Optional[str] = typer.Option(None),
    skip_preattack: bool = typer.Option(False, help="Skip pre-attack data expansion step"),
):
    """Finetuning-free end-to-end pipeline for one round."""
    if last_epoch is None:
        last_epoch = now_epoch - 1
    if backbone == "auto":
        backbone = "llama3"

    if not skip_preattack:
        preattack(
            base_dir=base_dir,
            last_epoch=last_epoch,
            target_backbone=backbone,
            attacker_checkpoint=attacker_checkpoint,
            target_checkpoint=target_checkpoint,
            provider=provider,
            api_provider=api_provider,
            api_model_id=api_model_id,
        )

    attack(
        base_dir=base_dir,
        epoch=last_epoch,
        attacker_checkpoint=attacker_checkpoint,
        provider=provider,
        api_provider=api_provider,
        api_model_id=api_model_id,
    )

    respond(
        base_dir=base_dir,
        epoch=last_epoch,
        backbone=backbone,
        target_checkpoint=target_checkpoint,
        provider=provider,
        api_provider=api_provider,
        api_model_id=api_model_id,
    )

    score(epoch=last_epoch, target_stage="pre_attack_target_stage1", base_dir=base_dir, model_role="target_pre-attack")
    select(now_epoch=now_epoch)


if __name__ == "__main__":
    app()