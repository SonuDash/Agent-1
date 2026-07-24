# Local Qwen Agent - Email, Calendar & Notion Assistant

A local, privacy-first agent powered by `qwen3.5:9b` running through Ollama on your Mac.
It can:

1. Read your Gmail and summarize today's **action items**
2. Create/list **Google Calendar** events on command
3. Search and read your **Notion** workspace for general questions

Nothing leaves your machine except the actual API calls to Google/Notion (which you'd
be making anyway) - the LLM inference itself is 100% local.

---

## 0. Prerequisites

- macOS on Apple Silicon (>=M3)
- Python 3.10+
- ~10GB free disk space

## 1. Install Ollama and pull the model

```bash
brew install ollama
ollama serve &          # starts the local server on http://localhost:11434
ollama pull qwen3.5:9b  # ~6.6GB download
```

Sanity check:
```bash
ollama run qwen3.5:9b "Say hello in one sentence."
```

## 2. Set up this project

```bash
cd qwen-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 3. Google Cloud setup (Gmail + Calendar)

1. Go to https://console.cloud.google.com/ → create a new project (e.g. "local-agent")
2. Enable **Gmail API** and **Google Calendar API** (APIs & Services → Library)
3. Configure the **OAuth consent screen** → User type: External → add yourself as a test user
4. Credentials → Create Credentials → **OAuth client ID** → Application type: **Desktop app**

5. **Build `credentials.json` by hand.** The current Console UI doesn't reliably offer
   a one-click JSON download (and it no longer lets you view an existing client secret
   after creation), so gather the pieces manually instead:
   - **Client ID** - shown on the client's detail page (click the client name, or the
     pencil/edit icon, from the Credentials list)
   - **Client secret** - if you can't see it (shown masked, e.g. `****o1c5`), click
     **Add Secret** on that same page to generate a new one. It's shown in full
     **only once** - copy it immediately
   - **Project ID** - click the project selector dropdown at the top of the Console
     (next to the Google Cloud logo), or go to **IAM & Admin → Settings**

   Then create `credentials.json` in the project root with this structure:

   ```json
   {
     "installed": {
       "client_id": "YOUR_CLIENT_ID.apps.googleusercontent.com",
       "project_id": "YOUR_PROJECT_ID",
       "auth_uri": "https://accounts.google.com/o/oauth2/auth",
       "token_uri": "https://oauth2.googleapis.com/token",
       "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
       "client_secret": "YOUR_CLIENT_SECRET",
       "redirect_uris": ["http://localhost"]
     }
   }
   ```

   (If your Console instance *does* show a working download button, just use that instead
   - it produces the same file.)

6. First run of the agent will pop open a browser window to authorize - this creates
   a `token.json` that's reused after that (no repeated logins)

Scopes used (edit `GOOGLE_SCOPES` in `config.py` if you want to change these):
- `gmail.readonly` - read email
- `gmail.compose` / `gmail.send` - draft and send email on your behalf
- `calendar` (read/write, needed to create/delete events)

**If you're updating from an earlier version of this project** that only had
`gmail.readonly`: delete `token.json` and re-run the agent so it re-authorizes
with the new scopes. Google won't silently grant new permissions to an
existing token - you'll get permission errors on drafting/sending until you
do this.

## 4. Notion setup

1. Go to https://www.notion.so/my-integrations → **New integration**
2. Name it (e.g. "Local Agent"), copy the **Internal Integration Token**
3. Paste it into `.env` as `NOTION_TOKEN`
4. **Important:** open each Notion page/database you want the agent to access →
   `···` menu → **Connections** → add your integration. Notion integrations only see
   pages explicitly shared with them.

## 5. Run it

Interactive agent (ad-hoc commands):
```bash
python agent.py
```
Example prompts:
- "What are my action items for today based on my email?"
- "Schedule a call with Priya tomorrow at 3pm for 30 minutes"
- "What does my Notion say about the Q3 roadmap?"

Daily briefing (standalone, good for a scheduled job):
```bash
python daily_briefing.py
```

## 6. Conversation history

Every query and response in `agent.py` is automatically logged to a local SQLite
database, `agent_log.db`, created in the project root on first run. No setup needed.

Why SQLite instead of a plain text log: it stays queryable as the log grows -
you can search by keyword or date instead of grepping through a flat file, and
it's still just one file, no server involved.

Query it directly from the terminal:
```bash
sqlite3 agent_log.db "SELECT timestamp, user_query FROM conversation_log ORDER BY id DESC LIMIT 10"
```

Or just ask the agent - it has a `search_conversation_log` tool wired in, so you
can say things like:
- "What did I ask you about calendar events last week?"
- "Show me my recent Notion queries"

The DB is gitignored (`agent_log.db`) since it's your personal query history,
not something to commit.

## 7. Automate the daily briefing (launchd)

Create `~/Library/LaunchAgents/com.local.qwenagent.briefing.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.local.qwenagent.briefing</string>
  <key>ProgramArguments</key>
  <array>
    <string>/absolute/path/to/qwen-agent/venv/bin/python</string>
    <string>/absolute/path/to/qwen-agent/daily_briefing.py</string>
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>7</integer>
    <key>Minute</key><integer>30</integer>
  </dict>
  <key>StandardOutPath</key><string>/tmp/qwenagent.log</string>
  <key>StandardErrorPath</key><string>/tmp/qwenagent.err</string>
</dict>
</plist>
```

Load it:
```bash
launchctl load ~/Library/LaunchAgents/com.local.qwenagent.briefing.plist
```

Every morning at 7:30am it'll write your briefing to `/tmp/qwenagent.log`. You can
swap the last line of `daily_briefing.py` for a `osascript` call to fire a native
macOS notification instead of just printing, if you want.

## Project layout

```
qwen-agent/
├── agent.py              # interactive CLI agent w/ tool-calling loop
├── daily_briefing.py     # standalone script: email → action items
├── storage.py             # SQLite logger for past queries/responses
├── config.py              # env/config loader
├── google_auth.py          # OAuth flow for Gmail + Calendar
├── tools/
│   ├── gmail_tools.py
│   ├── calendar_tools.py
│   └── notion_tools.py
├── requirements.txt
└── .env.example
```

## Notes / gotchas

- **Tool calling**: `qwen3.5:9b` supports native function calling through Ollama's
  `/api/chat` endpoint (OpenAI-compatible `tools` param). If you ever swap to a
  smaller/different model and tool calls stop firing reliably, that's the first
  thing to check.
- **Context window**: emails can be long. `gmail_tools.py` truncates each email body
  to ~1500 characters before handing it to the model - plenty for action-item
  extraction without blowing the context window.
- **Token refresh**: Google's `token.json` auto-refreshes; if it ever fully expires,
  just delete `token.json` and re-run to re-authorize.
