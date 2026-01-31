"""
End-to-end helper to fetch a Bhagavad Gita verse, render an Instagram-ready image,
build a caption, and (optionally) hand off to a poster client.

Env vars (.env):
  RAPIDAPI_KEY=<rapidapi_key>
  IG_ACCESS_TOKEN=<graph_api_access_token>  # for Graph API
  IG_USER_ID=<graph_user_id>
  IG_USERNAME/IG_PASSWORD  # for instagrapi (personal accounts)
  S3_BUCKET/S3_REGION/S3_ACCESS_KEY/S3_SECRET_KEY if you host media externally.

Usage (dry run, renders assets only):
  python scripts/post_geeta.py --background geeta.JPG [--font-path path/to/font.ttf] [--make-video --audio-path music.mp3]

Post (Graph API or instagrapi):
  python scripts/post_geeta.py --background geeta.JPG --make-video --post --uploader auto \
    [--video-url https://public.url/video.mp4]  # or supply S3_* envs to auto-upload
"""
import argparse
import json
import os
import random
import textwrap
import subprocess
import time
from datetime import datetime, UTC
from pathlib import Path
from typing import Optional, Tuple, Set, List

import requests
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont
from pydantic import ValidationError

# Load environment variables from .env if present
load_dotenv()
# Force temp directory to /tmp for Lambda/container environments
os.environ["TMPDIR"] = os.environ.get("TMPDIR", "/tmp")
os.environ["TEMP"] = os.environ.get("TEMP", "/tmp")
os.environ["TEMPDIR"] = os.environ.get("TEMPDIR", "/tmp")
SCRIPT_ROOT = Path(__file__).resolve().parent

RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "shreemad-bhagvad-geeta.p.rapidapi.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Instagram creds (choose Graph API or instagrapi)
IG_ACCESS_TOKEN = os.getenv("IG_ACCESS_TOKEN", "")
IG_USER_ID = os.getenv("IG_USER_ID", "")
IG_USERNAME = os.getenv("IG_USERNAME", "")
IG_PASSWORD = os.getenv("IG_PASSWORD", "")

# Optional S3 config if you decide to host media externally
S3_BUCKET = os.getenv("S3_BUCKET", "")
S3_REGION = os.getenv("S3_REGION", "")
S3_ACCESS_KEY = os.getenv("S3_ACCESS_KEY", "")
S3_SECRET_KEY = os.getenv("S3_SECRET_KEY", "")
HISTORY_S3_BUCKET = os.getenv("HISTORY_S3_BUCKET", "")

API_URL = "https://shreemad-bhagvad-geeta.p.rapidapi.com/shlokas/{chapter}/{verse}"

# Per-chapter verse counts to avoid invalid requests
VERSE_COUNTS = {
    1: 47,
    2: 72,
    3: 43,
    4: 42,
    5: 29,
    6: 47,
    7: 30,
    8: 28,
    9: 34,
    10: 42,
    11: 55,
    12: 20,
    13: 35,
    14: 27,
    15: 20,
    16: 24,
    17: 28,
    18: 78,
}

def choose_header(text: str) -> str:
    lowered = text.lower()
    if any(keyword in lowered for keyword in ["o lord", "lord krishna", "krishna", "madhusudan", "govinda"]):
        return "Arjun says"
    return "Krishna says"



def clean_brackets(text: str) -> str:
    # Remove common bracket characters to keep overlays clean.
    return text.replace("[", "").replace("]", "").replace("(", "").replace(")", "")


def ai_contextualize(verse_text: str, translation: str, model: str) -> Tuple[Optional[str], Optional[str]]:
    """Use OpenAI (if configured) to infer speaker and summarize as life teaching."""
    if not OPENAI_API_KEY:
        return None, None
    try:
        import openai
    except Exception:
        return None, None
    try:
        client = openai.OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None, None

    prompt = (
        "You are an expert editor. Given a Bhagavad Gita verse, do two things in English:\n"
        "1) Infer who is speaking (Krishna or Arjuna).\n"
        "2) Write a two-line inspirational summary (rich vocabulary, devotional, uplifting) of the dialogue/teaching.\n"
        "Respond as JSON with keys: speaker (Krishna/Arjuna) and teaching (two-line summary, using '\\n' for the break).\n"
        f"Verse (Sanskrit): {verse_text}\n"
        f"Translation (English): {translation}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        content = resp.choices[0].message.content
        data = json.loads(content)
        speaker = data.get("speaker")
        teaching = data.get("teaching")
        return speaker, teaching
    except Exception:
        return None, None


def pick_random_reference() -> Tuple[int, int]:
    chapter = random.randint(1, 18)
    verse = random.randint(1, VERSE_COUNTS[chapter])
    return chapter, verse


def load_history(history_path: Path) -> Set[Tuple[int, int]]:
    # If S3 history is configured, sync down first.
    if HISTORY_S3_BUCKET:
        try:
            import boto3
        except Exception:
            pass
        else:
            try:
                s3 = boto3.client(
                    "s3",
                    region_name=S3_REGION or None,
                    aws_access_key_id=S3_ACCESS_KEY or None,
                    aws_secret_access_key=S3_SECRET_KEY or None,
                )
                s3.download_file(HISTORY_S3_BUCKET, "posted_verses.json", str(history_path))
            except Exception:
                pass

    if not history_path.exists():
        return set()
    try:
        data = json.loads(history_path.read_text())
        return {tuple(item) for item in data if isinstance(item, list) and len(item) == 2}
    except Exception:
        return set()


def save_history(history_path: Path, history: Set[Tuple[int, int]]) -> None:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(sorted(list(history))))
    # Push back to S3 if configured.
    if HISTORY_S3_BUCKET:
        try:
            import boto3
        except Exception:
            return
        try:
            s3 = boto3.client(
                "s3",
                region_name=S3_REGION or None,
                aws_access_key_id=S3_ACCESS_KEY or None,
                aws_secret_access_key=S3_SECRET_KEY or None,
            )
            s3.upload_file(str(history_path), HISTORY_S3_BUCKET, "posted_verses.json")
        except Exception:
            pass


def fetch_verse(chapter: int, verse: int) -> dict:
    if not RAPIDAPI_KEY:
        raise RuntimeError("Missing RAPIDAPI_KEY env var.")

    url = API_URL.format(chapter=chapter, verse=verse)
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": RAPIDAPI_HOST,
    }
    resp = requests.get(url, headers=headers, timeout=20)
    # Gracefully handle common API failures (403/429/5xx) so we can fallback.
    if resp.status_code == 403:
        raise RuntimeError("RapidAPI returned 403 (forbidden). Check RAPIDAPI_KEY/host or rate limits.")
    if resp.status_code == 429:
        raise RuntimeError("RapidAPI rate limited (429).")
    resp.raise_for_status()

    if "application/json" not in resp.headers.get("Content-Type", ""):
        raise RuntimeError(f"Unexpected response type: {resp.headers.get('Content-Type')}")

    data = resp.json()
    # The API field names can vary; try a few sensible keys for the new endpoint.
    verse_text = (
        data.get("shloka")
        or data.get("sanskrit")
        or data.get("text")
        or data.get("shloka_text")
        or ""
    )
    translation = (
        data.get("en")
        or data.get("translation")
        or data.get("meaning")
        or data.get("english")
        or ""
    )
    verse_text = clean_brackets(verse_text)
    translation = clean_brackets(translation)
    if not verse_text:
        raise RuntimeError(f"Verse text missing in response: {data}")
    if not translation:
        translation = "No translation found."

    return {
        "chapter": chapter,
        "verse": verse,
        "verse_text": verse_text.strip(),
        "translation": translation.strip(),
        "raw": data,
    }


def wrap_text_by_width(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> str:
    """Wrap text based on rendered pixel width to improve fit."""
    if not text:
        return ""

    def measure(s: str) -> int:
        # Compatibility across Pillow versions
        if hasattr(draw, "textbbox"):
            bbox = draw.textbbox((0, 0), s, font=font)
            return bbox[2] - bbox[0]
        if hasattr(font, "getsize"):
            w, _ = font.getsize(s)
            return w
        # Fallback: approximate using textlength
        return int(draw.textlength(s, font=font))

    words = text.split()
    lines: List[str] = []
    current: List[str] = []
    for word in words:
        trial = " ".join(current + [word]) if current else word
        width = measure(trial)
        if width <= max_width:
            current.append(word)
        else:
            if current:
                lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return "\n".join(lines)


def load_font(font_path: Optional[Path], size: int) -> ImageFont.FreeTypeFont:
    # Try user-provided font, then common serif/script fonts, then default.
    candidates = []
    if font_path:
        candidates.append(font_path)
    candidates.extend(
        [
            Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
            Path("/System/Library/Fonts/Supplemental/Georgia.ttf"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            try:
                return ImageFont.truetype(str(candidate), size=size)
            except Exception:
                continue
    # Fallback to Pillow's default bitmap font
    return ImageFont.load_default()


def split_translation(text: str, threshold: int = 220) -> List[str]:
    """Split long text into two parts by words to fit into two segments."""
    words = text.split()
    if len(text) <= threshold or len(words) <= 8:
        return [text]
    total_len = len(text)
    half_target = total_len // 2
    current = []
    current_len = 0
    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= half_target:
            break
    part1 = " ".join(current).strip()
    part2 = " ".join(words[len(current) :]).strip()
    return [part1, part2] if part2 else [text]


def render_image(
    background: Path,
    header: str,
    translation: str,
    output_path: Path,
    font_path: Optional[Path],
    translation_size: int,
    footer_text: Optional[str] = None,
) -> Path:
    # Render a vertical image with dark overlay for contrast.
    bg = Image.open(background).convert("RGBA").resize((1080, 1920))
    overlay = Image.new("RGBA", bg.size, (0, 0, 0, 140))
    composed = Image.alpha_composite(bg, overlay)
    draw = ImageDraw.Draw(composed)

    W, H = composed.size
    max_width = int(W * 0.82)  # pixels for text block

    def measure_fonts(body_size: int) -> Tuple[int, dict]:
        body_font = load_font(font_path, body_size)
        header_font = load_font(font_path, max(body_size - 10, 50))
        wrapped_header = wrap_text_by_width(header, header_font, max_width, draw)
        wrapped_translation = wrap_text_by_width(translation, body_font, max_width, draw)
        header_bbox = draw.multiline_textbbox((0, 0), wrapped_header, font=header_font, spacing=10, align="center")
        translation_bbox = draw.multiline_textbbox((0, 0), wrapped_translation, font=body_font, spacing=10, align="center")
        header_w = header_bbox[2] - header_bbox[0]
        header_h = header_bbox[3] - header_bbox[1]
        translation_w = translation_bbox[2] - translation_bbox[0]
        translation_h = translation_bbox[3] - translation_bbox[1]
        gap = 40
        total_h = header_h + gap + translation_h
        return total_h, {
            "body_font": body_font,
            "header_font": header_font,
            "wrapped_header": wrapped_header,
            "wrapped_translation": wrapped_translation,
            "header_w": header_w,
            "header_h": header_h,
            "translation_w": translation_w,
            "translation_h": translation_h,
            "gap": gap,
        }

    available_h = int(H * 0.8)
    body_size = translation_size
    measured_h, parts = measure_fonts(body_size)
    while measured_h > available_h and body_size > 30:
        body_size -= 2
        measured_h, parts = measure_fonts(body_size)

    header_x = (W - parts["header_w"]) // 2
    translation_x = (W - parts["translation_w"]) // 2
    start_y = max(int((H - measured_h) * 0.3), 60)

    draw.multiline_text((header_x, start_y), parts["wrapped_header"], fill="white", font=parts["header_font"], spacing=10, align="center")
    draw.multiline_text(
        (translation_x, start_y + parts["header_h"] + parts["gap"]),
        parts["wrapped_translation"],
        fill="white",
        font=parts["body_font"],
        spacing=10,
        align="center",
    )

    if footer_text:
        footer_font = load_font(font_path, max(body_size // 3, 30))
        footer_bbox = draw.multiline_textbbox((0, 0), footer_text, font=footer_font, spacing=8, align="center")
        footer_w = footer_bbox[2] - footer_bbox[0]
        footer_x = (W - footer_w) // 2
        footer_y = H - footer_bbox[3] - 60
        draw.multiline_text((footer_x, footer_y), footer_text, fill="white", font=footer_font, spacing=8, align="center")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    composed.convert("RGB").save(output_path)
    return output_path


def make_video_ffmpeg(
    image_path: Path,
    audio_path: Optional[Path],
    output_path: Path,
    duration: float = 10.0,
    fps: int = 30,
    zoom_strength: float = 0.0,
) -> Path:
    """
    Render a video using ffmpeg directly to avoid MoviePy temp path issues.
    """
    img_abs = str(image_path.absolute())
    out_abs = str(output_path.absolute())
    audio_abs = str(audio_path.absolute()) if audio_path else None

    # Simple zoom effect handled by scaling over time via fps/pad if desired
    zoom_filter = ""
    if zoom_strength and zoom_strength > 0:
        # light zoom over duration
        zoom_filter = f",zoompan=z='min(zoom+{zoom_strength}/({duration}*{fps}),1.1)':d=1:fps={fps}"

    vf = f"fps={fps},scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2{zoom_filter}"

    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-i",
        img_abs,
    ]
    if audio_abs:
        cmd += ["-i", audio_abs]
    cmd += [
        "-t",
        str(duration),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
    ]
    if audio_abs:
        cmd += ["-c:a", "aac", "-shortest"]
    cmd.append(out_abs)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)
    return output_path


def upload_to_s3(file_path: Path) -> str:
    if not (S3_BUCKET and S3_REGION and S3_ACCESS_KEY and S3_SECRET_KEY):
        raise RuntimeError("S3 config missing. Set S3_BUCKET, S3_REGION, S3_ACCESS_KEY, S3_SECRET_KEY.")
    try:
        import boto3
    except Exception as exc:
        raise RuntimeError("boto3 is required for S3 upload. Install with `pip install boto3`.") from exc

    object_name = file_path.name
    s3 = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=S3_ACCESS_KEY,
        aws_secret_access_key=S3_SECRET_KEY,
    )
    extra_args = {"ACL": "public-read", "ContentType": "video/mp4" if file_path.suffix.lower() == ".mp4" else "image/png"}
    s3.upload_file(str(file_path), S3_BUCKET, object_name, ExtraArgs=extra_args)
    return f"https://{S3_BUCKET}.s3.{S3_REGION}.amazonaws.com/{object_name}"

def build_caption(info: dict) -> str:
    verse_ref = f"BG {info['chapter']}:{info['verse']}"
    base = (
        f"✨ Bhagavad Gita • {verse_ref}\n\n"
        f"Sanskrit: {info.get('verse_text', '')}\n\n"
        f"{info['translation']}\n\n"
    )
    if info.get("teaching"):
        base += f"Life teaching: {info['teaching']}\n\n"
    base += "#BhagavadGita #SpiritualWisdom #InnerPeace #Meditation #Yoga #SanatanDharma"
    return base


def post_to_instagram_graph(media_url: str, caption: str) -> None:
    if not (IG_ACCESS_TOKEN and IG_USER_ID):
        raise RuntimeError("Graph API posting requires IG_ACCESS_TOKEN and IG_USER_ID.")

    # Create media container
    create_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media"
    payload = {"caption": caption, "access_token": IG_ACCESS_TOKEN}
    if media_url.lower().endswith((".mp4", ".mov")):
        payload["media_type"] = "VIDEO"
        payload["video_url"] = media_url
    else:
        payload["image_url"] = media_url
    resp = requests.post(create_url, data=payload, timeout=30)
    resp.raise_for_status()
    creation_id = resp.json().get("id")
    if not creation_id:
        raise RuntimeError(f"Graph API did not return creation id: {resp.text}")

    # Poll status
    status_url = f"https://graph.facebook.com/v19.0/{creation_id}"
    for _ in range(10):
        status_resp = requests.get(status_url, params={"fields": "status_code", "access_token": IG_ACCESS_TOKEN}, timeout=15)
        status_resp.raise_for_status()
        status_code = status_resp.json().get("status_code")
        if status_code == "FINISHED":
            break
        if status_code == "ERROR":
            raise RuntimeError(f"Graph API media processing failed: {status_resp.text}")
        time.sleep(3)
    else:
        raise RuntimeError("Graph API media processing did not finish in time.")

    # Publish
    publish_url = f"https://graph.facebook.com/v19.0/{IG_USER_ID}/media_publish"
    publish_resp = requests.post(publish_url, data={"creation_id": creation_id, "access_token": IG_ACCESS_TOKEN}, timeout=30)
    publish_resp.raise_for_status()


def post_to_instagram_instagrapi(media_path: Path, caption: str) -> None:
    if not (IG_USERNAME and IG_PASSWORD):
        raise RuntimeError("Instagrapi posting requires IG_USERNAME and IG_PASSWORD.")
    try:
        from instagrapi import Client
    except Exception as exc:
        raise RuntimeError("instagrapi is required for this mode. Install with `pip install instagrapi`.") from exc

    session_path = Path("artifacts/ig_session.json")
    session_path.parent.mkdir(parents=True, exist_ok=True)

    cl = Client()

    if session_path.exists():
        try:
            cl.load_settings(str(session_path))
            cl.login(IG_USERNAME, IG_PASSWORD)
        except Exception:
            cl = Client()
            cl.login(IG_USERNAME, IG_PASSWORD)
    else:
        cl.login(IG_USERNAME, IG_PASSWORD)

    try:
        cl.dump_settings(str(session_path))
    except Exception:
        pass

    if media_path.suffix.lower() in {".mp4", ".mov"}:
        try:
            cl.clip_upload(str(media_path), caption)
        except ValidationError:
            return
    else:
        cl.photo_upload(str(media_path), caption)


def post_to_instagram(media_path: Path, caption: str, media_url: Optional[str] = None, uploader: str = "auto") -> None:
    uploader = uploader.lower()
    if uploader not in {"auto", "graph", "instagrapi"}:
        raise ValueError("uploader must be one of: auto, graph, instagrapi.")

    if uploader in {"auto", "graph"} and IG_ACCESS_TOKEN and IG_USER_ID:
        if not media_url:
            raise RuntimeError("Graph API requires a publicly accessible media_url. Provide --video-url or enable S3 upload.")
        post_to_instagram_graph(media_url, caption)
        return

    if uploader in {"auto", "instagrapi"} and IG_USERNAME and IG_PASSWORD:
        post_to_instagram_instagrapi(media_path, caption)
        return

    raise RuntimeError("No valid Instagram credentials found. Set IG_ACCESS_TOKEN/IG_USER_ID or IG_USERNAME/IG_PASSWORD.")


def main():
    parser = argparse.ArgumentParser(description="Create a Gita verse post (image + caption).")
    parser.add_argument("--background", type=Path, default=Path("geeta.JPG"), help="Background image path")
    parser.add_argument("--output-dir", type=Path, default=Path(os.getenv("OUTPUT_DIR", "/tmp/artifacts")), help="Where to write outputs")
    default_history = Path(os.getenv("HISTORY_PATH", "artifacts/posted_verses.json"))
    parser.add_argument("--history-file", type=Path, default=default_history, help="Tracks posted chapter/verse pairs to avoid repeats")
    parser.add_argument("--post", action="store_true", help="If set, attempt to post to Instagram")
    parser.add_argument("--font-path", type=Path, default=None, help="Path to a .ttf/.otf font file")
    parser.add_argument("--verse-font-size", type=int, default=60, help="Font size for verse text")
    parser.add_argument("--translation-font-size", type=int, default=100, help="Font size for translation")
    parser.add_argument("--make-video", action="store_true", help="Create a short video from the rendered image")
    parser.add_argument("--audio-path", type=Path, default=None, help="Optional background audio for the video")
    parser.add_argument("--video-duration", type=float, default=10.0, help="Video duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Video frames per second")
    parser.add_argument("--zoom-strength", type=float, default=0.0, help="Zoom factor over the video duration (0 for static)")
    parser.add_argument("--video-url", type=str, default=None, help="Publicly accessible media URL (for Graph API posting)")
    parser.add_argument("--uploader", type=str, default="auto", help="Posting client: auto|graph|instagrapi")
    parser.add_argument("--manual-translation", type=str, default=None, help="Use provided translation text (skip API)")
    parser.add_argument("--footer-text", type=str, default="#chantthebhagvadgeeta", help="Footer/hashtag text")
    parser.add_argument("--reel", action="store_true", help="Shortcut to render a video (reel) with audio if available")
    parser.add_argument("--ai-context", action="store_true", default=True, help="Use OpenAI (if configured) to infer speaker and teaching")
    parser.add_argument("--ai-model", type=str, default="gpt-3.5-turbo", help="OpenAI model to use for context/teaching")
    args = parser.parse_args()

    # If reel is requested, ensure we produce a video.
    if args.reel:
        args.make_video = True

    if not args.background.exists():
        raise FileNotFoundError(f"Background image not found: {args.background}")

    history = load_history(args.history_file)
    available_refs = {(c, v) for c in VERSE_COUNTS for v in range(1, VERSE_COUNTS[c] + 1)}
    remaining = available_refs - history
    if not remaining:
        raise RuntimeError("No remaining verses available. Reset or clear your history file to continue.")

    chapter, verse = random.choice(list(remaining))
    fallback_translation = os.getenv("FALLBACK_TRANSLATION", None)

    if args.manual_translation:
        info = {
            "chapter": chapter,
            "verse": verse,
            "verse_text": "",
            "translation": clean_brackets(args.manual_translation.strip()),
            "raw": {},
        }
    else:
        try:
            info = fetch_verse(chapter, verse)
        except Exception as exc:
            if fallback_translation:
                info = {
                    "chapter": chapter,
                    "verse": verse,
                    "verse_text": "",
                    "translation": clean_brackets(fallback_translation.strip()),
                    "raw": {"error": str(exc)},
                }
                print(f"Fetch failed ({exc}); using FALLBACK_TRANSLATION.")
            else:
                raise


    speaker_ai = None
    teaching_ai = None
    if args.ai_context:
        speaker_ai, teaching_ai = ai_contextualize(info.get("verse_text", ""), info["translation"], args.ai_model)

    translation_lower = info["translation"].strip().lower()
    if translation_lower.startswith(("arjuna said", "arjun said", "krishna said", "lord krishna said", "o krishna")):
        header_text = ""
    else:
        header_text = f"{(speaker_ai or choose_header(info['translation'])).strip()}".rstrip(":")
    translation_parts = split_translation(info["translation"])
    if teaching_ai:
        info["teaching"] = teaching_ai
    merged_translation_for_image = "\n\n".join(translation_parts)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir
    image_path = out_dir / f"gita_{chapter}_{verse}_{timestamp}.png"
    json_path = out_dir / f"gita_{chapter}_{verse}_{timestamp}.json"

    render_image(
        args.background,
        header_text,
        merged_translation_for_image,
        image_path,
        args.font_path,
        args.translation_font_size,
        footer_text=args.footer_text,
    )
    caption = build_caption(info)

    video_path = None
    if args.make_video:
        video_path = out_dir / f"gita_{chapter}_{verse}_{timestamp}.mp4"
        audio_path = args.audio_path
        default_candidates = [
            SCRIPT_ROOT / "scripts" / "music.mp3",
            SCRIPT_ROOT / "music.mp3",
            Path("music.mp3"),
            Path("AUDIO-2025-06-22-18-36-55.mp3"),
        ]
        default_audio = next((p for p in default_candidates if p.exists()), None)
        if audio_path is None and default_audio and default_audio.exists():
            audio_path = default_audio

        make_video_ffmpeg(
            image_path=image_path,
            audio_path=audio_path,
            output_path=video_path,
            duration=args.video_duration,
            fps=args.fps,
            zoom_strength=args.zoom_strength,
        )

    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "chapter": info["chapter"],
        "verse": info["verse"],
        "translation": info["translation"],
        "teaching": info.get("teaching"),
        "source": info["raw"],
        "caption": caption,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))

    print(f"Rendered image: {image_path}")
    if video_path:
        print(f"Rendered video: {video_path}")
    print(f"Saved metadata: {json_path}")
    print("Caption:\n", caption)

    if args.post:
        media_path = video_path if video_path else image_path
        media_url = args.video_url
        # Only upload to S3 when using Graph API (auto will prefer graph if creds+url exist).
        if args.uploader in {"auto", "graph"}:
            if not media_url and (S3_BUCKET and S3_REGION and S3_ACCESS_KEY and S3_SECRET_KEY):
                media_url = upload_to_s3(media_path)
        post_to_instagram(media_path, caption, media_url=media_url, uploader=args.uploader)
        # Record posted verse to avoid repeats.
        history.add((info["chapter"], info["verse"]))
        save_history(args.history_file, history)
        print("Posted to Instagram.")
    else:
        print("Dry run complete (no Instagram post).")


if __name__ == "__main__":
    main()
