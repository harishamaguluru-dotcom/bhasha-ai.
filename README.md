# Bhasha AI – Local Language Helper

> **Understand Every Word, In Your Language.**

Bhasha AI helps people who are not comfortable with English understand it in their own Indian language. Paste or speak English, and get it back in **Telugu, Hindi, Tamil, Kannada, Malayalam or Marathi**, either as a faithful translation or as a simple explanation in everyday words.

Telugu is the priority language and the default everywhere.

## Features

| Feature | How it works |
|---|---|
| Direct Translation | Gemini translates faithfully, keeping names, dates, numbers and amounts exactly as written. |
| Simple Explanation | Gemini explains the meaning in short sentences and easy words, with an example when useful. |
| Voice input | Record with the microphone; Gemini transcribes it (English or any supported Indian language) and can translate straight away. |
| Listen | Google Text-to-Speech reads the result aloud in all six languages. |
| Copy | One-click copy of the result. |
| Pages | Home, Translate, Explain in Simple Language, About Project. |

## Project structure

```
app.py            Streamlit UI: landing page, pages, theme, voice and listen flows
translator.py     Languages, validation, prompts, Gemini calls, friendly errors
speech.py         Speech-to-text (Gemini) and text-to-speech (gTTS)
requirements.txt  Python dependencies
tests/            Offline unit tests (Gemini and gTTS are mocked)
.streamlit/       Theme and server settings
```

## Setup on Replit

1. Create a new Repl (Python) and upload these files, or import this folder.
2. Get a free Gemini API key at <https://aistudio.google.com/apikey>.
3. In Replit open **Tools → Secrets** (the lock icon) and add:
   - **Key:** `GEMINI_API_KEY`
   - **Value:** your API key
4. Install packages: `pip install -r requirements.txt` (Replit usually does this automatically).
5. Set the run command to:
   ```
   streamlit run app.py --server.port 5000 --server.address 0.0.0.0
   ```
6. Press **Run**. Replit shows the app in the Webview.

Secrets are exposed to the app as environment variables on the server. The key is never written into code and never sent to the browser. If you add or change a secret while the app is running, stop and run it again.

## Setup on your own computer

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then put your key in .env
streamlit run app.py
```

You can also export the variable directly: `export GEMINI_API_KEY="your-key"`.

### Optional settings

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | Gemini API key | none (required) |
| `GEMINI_MODEL` | Gemini model to use | `gemini-2.5-flash` |

## Using the app

1. Open **Translate** (or **Explain in Simple Language**).
2. Type or paste English, tap one of the example buttons, or press **Voice input** and speak.
3. Choose the language (Telugu is pre-selected) and the mode.
4. Press **Translate** / **Explain simply**.
5. Use **Copy translation** or **Listen** under the result.

## Testing

Offline unit tests (no key or internet needed):

```bash
python -m unittest discover -s tests -v
```

They cover input validation, prompts for every language and mode, Markdown cleanup, missing key, unsupported languages, connection failures with retries, empty AI answers, transcription and text-to-speech checks.

To test with the real services, add your key and try this checklist in the app:

- [ ] Translate a sentence into Telugu, then each of the other five languages.
- [ ] Switch to Simple Explanation and explain a sentence with a number and a date; confirm they are unchanged.
- [ ] Press Voice input, say a sentence, and confirm it appears and is translated.
- [ ] Press Listen for Telugu, Hindi and Tamil.
- [ ] Press Copy and paste somewhere.
- [ ] Try empty input, a text with only numbers, and the app without a key; each should show a clear message.

## Error handling

| Situation | What the user sees |
|---|---|
| Empty or symbol-only text | A message asking for English words |
| Text over 5,000 characters | A message to shorten or split it |
| No API key | Steps to add `GEMINI_API_KEY` (also shown in the sidebar) |
| Rejected key, quota used up, busy server | A specific plain-language message; busy servers are retried automatically |
| No internet | A message to check the connection |
| Unsupported spoken or voice language | A message listing supported languages |
| No speech in a recording | A message to speak closer and try again |

## Troubleshooting

- **"API key missing"** – check the secret name is exactly `GEMINI_API_KEY`, then restart the app.
- **Microphone does nothing** – allow microphone access in the browser. Recording needs HTTPS or `localhost` (Replit provides HTTPS).
- **Listen fails** – gTTS needs internet access to Google's voice service.
- **Model not found** – set `GEMINI_MODEL` to a model your key can use.
- **Telugu looks like boxes** – your device lacks a Telugu font; the app loads Noto Sans fonts from Google Fonts when online.

## Privacy and limits

Text you enter is sent to Google's Gemini API for translation and to Google Text-to-Speech for audio. Do not enter passwords, bank PINs or other secrets. AI can make mistakes; for legal, medical or financial matters, confirm important details with a qualified person.
