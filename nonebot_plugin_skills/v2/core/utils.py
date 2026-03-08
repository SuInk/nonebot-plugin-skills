import base64
import uuid
from pathlib import Path
from typing import List, Optional, Union
from urllib.parse import unquote, urlparse

from nonebot import logger
from nonebot.adapters.onebot.v11 import Message, MessageSegment


def _image_output_dir() -> Path:
    path = Path("data/nonebot_plugin_skills/generated_images")
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_image_to_local(image_bytes: bytes, *, suffix: str = ".jpg") -> str:
    """
    Persist image bytes locally and return a CQ image tag using a file URI.
    This keeps message payloads small and works for recalled-message persistence.
    """
    output_path = _image_output_dir() / f"{uuid.uuid4().hex}{suffix}"
    output_path.write_bytes(image_bytes)
    return f"[CQ:image,file={output_path.resolve().as_uri()}]"


def _decode_base64_payload(payload: str) -> Optional[bytes]:
    try:
        return base64.b64decode(payload)
    except Exception as e:
        logger.error(f"Failed to decode base64 image payload: {e}")
        return None


def extract_image_sources(message: Union[str, Message]) -> List[str]:
    parsed = message if isinstance(message, Message) else Message(str(message))
    sources: List[str] = []
    for seg in parsed:
        if seg.type != "image":
            continue
        file_val = seg.data.get("file") or seg.data.get("url")
        if file_val:
            sources.append(file_val)
    return sources


def local_path_from_file_uri(uri: str) -> Optional[Path]:
    if not uri.startswith("file://"):
        return None

    parsed = urlparse(uri)
    path = unquote(parsed.path or "")
    if not path:
        return None

    if len(path) >= 3 and path[0] == "/" and path[2] == ":":
        path = path[1:]
    return Path(path)


async def build_image_segment(result: Union[str, bytes, MessageSegment]) -> Optional[MessageSegment]:
    """
    Convert common image outputs into a OneBot image segment.
    """
    if not result:
        return None

    if isinstance(result, MessageSegment):
        return result

    if isinstance(result, bytes):
        return MessageSegment.image(result)

    res_str = str(result).strip()
    if not res_str:
        return None

    if res_str.startswith("http://") or res_str.startswith("https://"):
        return MessageSegment.image(res_str)

    if res_str.startswith("file://"):
        return MessageSegment.image(res_str)

    if res_str.startswith("base64://"):
        payload = _decode_base64_payload(res_str[len("base64://") :])
        return MessageSegment.image(payload) if payload else None

    if res_str.startswith("data:image") and "," in res_str:
        payload = _decode_base64_payload(res_str.split(",", 1)[1])
        return MessageSegment.image(payload) if payload else None

    if res_str.startswith("[CQ:image,"):
        parsed = Message(res_str)
        for seg in parsed:
            if seg.type == "image":
                return seg

    if len(res_str) > 100:
        payload = _decode_base64_payload(res_str)
        if payload:
            return MessageSegment.image(payload)

    return None
