# Chant the Bhagavad Geeta – Daily Instagram Reel Automation

Automates a daily Bhagavad Gita reel: fetches a verse, renders an image/video with a background and audio, and posts to Instagram. Runs locally on macOS via `launchd` and can be deployed to AWS (Lambda/ECS) if needed.

## Features
- Verse fetch from RapidAPI (fallback text if API fails).
- Renders 1080x1920 image + short MP4 with background audio.
- Posts via Instagram (instagrapi) with idempotency options in AWS version.
- macOS daily schedule (8:00 PM local) via `launchd`.

## Local Usage
```bash
cd /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt instagrapi

# Run with custom background/audio; uses fallback translation if API fails
.venv/bin/python post_geeta.py \
  --background /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/geeta.JPG \
  --make-video \
  --audio-path /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/music.mp3 \
  --post --uploader instagrapi
```

### Environment / .env
- `IG_USERNAME`, `IG_PASSWORD` (instagrapi)
- `INSTAGRAM_ACCESS_TOKEN`, `INSTAGRAM_USER_ID` (Graph API if you switch uploaders)
- `RAPIDAPI_KEY`, `RAPIDAPI_HOST=shreemad-bhagvad-geeta.p.rapidapi.com`
- `S3_BUCKET_NAME` (if using S3 uploads/presign)
- Optional: `FALLBACK_TRANSLATION` to keep posting when API fails.

## macOS Schedule (8 PM daily)
LaunchAgent: `~/Library/LaunchAgents/com.geeta.poster.plist`
```xml
<key>StartCalendarInterval</key>
<dict><key>Hour</key><integer>20</integer><key>Minute</key><integer>0</integer></dict>
<key>ProgramArguments</key>
<array>
  <string>/bin/zsh</string>
  <string>-lc</string>
  <string>cd /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts && /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/.venv/bin/python /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/post_geeta.py --background /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/geeta.JPG --make-video --audio-path /Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/music.mp3 --post --uploader instagrapi</string>
 </array>
```
Load/start:
```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.geeta.poster.plist 2>/dev/null
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.geeta.poster.plist
launchctl kickstart -k gui/$(id -u)/com.geeta.poster   # optional immediate run
tail -f /tmp/geeta_poster.out /tmp/geeta_poster.err
```

## Background & Audio
- Background: `/Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/geeta.JPG`
- Audio: `/Users/ritzz/Desktop/ChantthebhagvadGeeta/scripts/music.mp3`
- Adjust paths in the command/LaunchAgent as needed.

## Known Limitations
- Requires network access to RapidAPI and `i.instagram.com`; if DNS blocks Instagram, switch to Graph API or fix DNS.
- RapidAPI may rate-limit; set `FALLBACK_TRANSLATION` to avoid failed posts.

## AWS (optional)
- Container/Lambda and ECS Fargate templates are in `infra/`. Use SAM/ECR for production scheduling if you prefer cloud execution.

