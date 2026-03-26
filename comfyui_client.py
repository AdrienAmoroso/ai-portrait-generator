"""ComfyUI API client — SDXL generation with optional ControlNet OpenPose."""

import hashlib
import json
import logging
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from config import (
    CFG_SCALE,
    COMFYUI_CHECKPOINT,
    COMFYUI_CONTROLNET_MODEL,
    COMFYUI_URL,
    CONTROLNET_STRENGTH,
    GENERATION_STEPS,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    POSES_DIR,
)

logger = logging.getLogger(__name__)


def pick_pose(player_id: int, gender: str = "") -> Path | None:
    """Deterministically pick a pose image for a player.

    Looks in ``poses/men/`` or ``poses/women/`` based on *gender*,
    falling back to ``poses/`` if the gender-specific subfolder is
    empty or missing.

    Returns None if no pose images are available.
    """
    if not POSES_DIR.is_dir():
        return None

    _IMG_EXT = {".png", ".jpg", ".jpeg"}

    # Try gender-specific subfolder first
    gender_dir = None
    g = gender.strip().upper()
    if g in {"MAN", "MALE", "M"}:
        gender_dir = POSES_DIR / "men"
    elif g in {"WOMAN", "FEMALE", "F"}:
        gender_dir = POSES_DIR / "women"

    poses: list[Path] = []
    if gender_dir and gender_dir.is_dir():
        poses = sorted(
            p for p in gender_dir.iterdir()
            if p.suffix.lower() in _IMG_EXT
        )

    # Fallback: root poses directory
    if not poses:
        poses = sorted(
            p for p in POSES_DIR.iterdir()
            if p.suffix.lower() in _IMG_EXT
        )

    if not poses:
        return None

    digest = hashlib.md5(
        f"{player_id}:pose".encode(), usedforsecurity=False
    ).hexdigest()
    index = int(digest, 16) % len(poses)
    return poses[index]


def _build_workflow(
    positive_prompt: str,
    negative_prompt: str,
    pose_image_path: Path | None,
    seed: int | None = None,
) -> dict:
    """Build a ComfyUI API workflow dict for SDXL + optional ControlNet."""
    if seed is None:
        seed = random.randint(0, 2**31)

    # Base nodes present in all workflows
    workflow: dict = {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": GENERATION_STEPS,
                "cfg": CFG_SCALE,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0,
                "model": ["4", 0] if pose_image_path is None else ["10", 0],
                "positive": ["6", 0],
                "negative": ["7", 0],
                "latent_image": ["5", 0],
            },
        },
        "4": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": COMFYUI_CHECKPOINT},
        },
        "5": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": IMAGE_WIDTH,
                "height": IMAGE_HEIGHT,
                "batch_size": 1,
            },
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": positive_prompt,
                "clip": ["4", 1],
            },
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": negative_prompt,
                "clip": ["4", 1],
            },
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["3", 0],
                "vae": ["4", 2],
            },
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {
                "images": ["8", 0],
                "filename_prefix": "player",
            },
        },
    }

    # Add ControlNet nodes if a pose image is available
    if pose_image_path is not None:
        workflow["10"] = {
            "class_type": "ControlNetLoader",
            "inputs": {"control_net_name": COMFYUI_CONTROLNET_MODEL},
        }
        workflow["11"] = {
            "class_type": "LoadImage",
            "inputs": {"image": pose_image_path.name},
        }
        workflow["12"] = {
            "class_type": "ControlNetApplyAdvanced",
            "inputs": {
                "strength": CONTROLNET_STRENGTH,
                "start_percent": 0.0,
                "end_percent": 1.0,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "control_net": ["10", 0],
                "image": ["11", 0],
            },
        }
        # Re-wire: sampler reads model from checkpoint, but conditioning from ControlNet
        workflow["3"]["inputs"]["model"] = ["4", 0]
        workflow["3"]["inputs"]["positive"] = ["12", 0]
        workflow["3"]["inputs"]["negative"] = ["12", 1]

    return workflow


def _upload_pose_image(pose_path: Path) -> None:
    """Upload a pose image to ComfyUI's input directory."""
    url = f"{COMFYUI_URL}/upload/image"
    boundary = "----FormBoundary" + hashlib.md5(
        str(time.time()).encode(), usedforsecurity=False
    ).hexdigest()[:16]

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="image"; filename="{pose_path.name}"\r\n'
        f"Content-Type: image/png\r\n\r\n"
    ).encode() + pose_path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        resp.read()


def _queue_prompt(workflow: dict) -> str:
    """Submit a workflow to ComfyUI and return the prompt_id."""
    payload = json.dumps({"prompt": workflow}).encode()
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["prompt_id"]


def _poll_until_done(prompt_id: str, timeout: int = 300) -> dict:
    """Poll ComfyUI history until the prompt completes or times out."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            url = f"{COMFYUI_URL}/history/{prompt_id}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                history = json.loads(resp.read())
            if prompt_id in history:
                return history[prompt_id]
        except urllib.error.URLError:
            pass
        time.sleep(1.0)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} did not complete within {timeout}s")


def _download_image(filename: str, subfolder: str, output_path: Path) -> None:
    """Download a generated image from ComfyUI."""
    params = urllib.parse.urlencode(
        {"filename": filename, "subfolder": subfolder, "type": "output"}
    )
    url = f"{COMFYUI_URL}/view?{params}"
    with urllib.request.urlopen(url, timeout=30) as resp:
        data = resp.read()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(data)


def generate_comfyui(
    positive_prompt: str,
    negative_prompt: str,
    output_path: Path,
    player_id: int,
    seed: int | None = None,
    gender: str = "",
) -> bool:
    """Generate an image with ComfyUI and save to *output_path*.

    Returns True on success, False on failure.
    """

    try:
        pose_path = pick_pose(player_id, gender)
        if pose_path is not None:
            logger.info("Using pose reference: %s", pose_path.name)
            _upload_pose_image(pose_path)

        workflow = _build_workflow(positive_prompt, negative_prompt, pose_path, seed)
        prompt_id = _queue_prompt(workflow)
        logger.info("Queued ComfyUI prompt %s", prompt_id)

        history = _poll_until_done(prompt_id)

        # Find the SaveImage output
        outputs = history.get("outputs", {})
        for node_id, node_output in outputs.items():
            images = node_output.get("images", [])
            for img_info in images:
                _download_image(
                    img_info["filename"],
                    img_info.get("subfolder", ""),
                    output_path,
                )
                logger.info("Saved %s", output_path)
                return True

        logger.error("No image output found for prompt %s", prompt_id)
        return False

    except Exception:
        logger.exception("ComfyUI generation failed for %s", output_path.name)
        return False
