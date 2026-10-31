"""Bhasha AI – Local Language Helper (Streamlit front end).

Run with:  streamlit run app.py
The Gemini API key is read on the server from the GEMINI_API_KEY secret /
environment variable and is never sent to the browser.
"""

from __future__ import annotations

import html
import json
import logging

import streamlit as st
import streamlit.components.v1 as components

from speech import SPEECH_LANGUAGES, synthesize_speech, transcribe_audio
from translator import (
    LANGUAGES,
    MAX_INPUT_CHARS,
    MISSING_KEY_MESSAGE,
    MODE_DIRECT,
    MODE_SIMPLE,
    MODES,
    BhashaError,
    get_api_key,
    get_model_name,
    translate_text,
)

logger = logging.getLogger("bhasha.app")

st.set_page_config(
    page_title="Bhasha AI – Local Language Helper",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Theme: white + light purple, rounded surfaces
# --------------------------------------------------------------------------- #

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Noto+Sans+Telugu:wght@400;500;600&family=Noto+Sans+Devanagari:wght@400;500;600&family=Noto+Sans+Tamil:wght@400;500;600&family=Noto+Sans+Kannada:wght@400;500;600&family=Noto+Sans+Malayalam:wght@400;500;600&display=swap');

:root {
  --ink: #241B4B; --muted: #6B6590; --violet: #7B5CF0; --violet-deep: #5A3FD1;
  --lav: #F3EFFF; --lav-2: #E7DFFF; --line: #E0D7FA; --white: #FFFFFF;
}
.stApp {
  font-family: 'Plus Jakarta Sans','Noto Sans Telugu','Noto Sans Devanagari','Noto Sans Tamil',
               'Noto Sans Kannada','Noto Sans Malayalam',system-ui,sans-serif;
  color: var(--ink);
  background:
    radial-gradient(900px 420px at 88% -8%, #E9DEFF 0%, rgba(233,222,255,0) 70%),
    radial-gradient(700px 380px at -5% 30%, #F4EEFF 0%, rgba(244,238,255,0) 70%),
    #FFFFFF;
}
header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { max-width: 1080px; padding-top: 2.2rem; padding-bottom: 4rem; }

h1, h2, h3 { color: var(--ink); letter-spacing: -0.02em; font-weight: 800; }
textarea, input, button { font-family: inherit !important; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #F0EAFF 0%, #FFFFFF 85%);
  border-right: 1px solid var(--line);
}
a[data-testid="stSidebarNavLink"] { border-radius: 14px; font-weight: 600; }
a[data-testid="stSidebarNavLink"][aria-current="page"] { background: var(--lav-2); color: var(--violet-deep); }

/* Brand */
.brand { display: flex; align-items: center; gap: .6rem; margin-bottom: 1rem; }
.brand-mark {
  width: 2.3rem; height: 2.3rem; border-radius: .8rem; display: grid; place-items: center;
  background: linear-gradient(135deg, #9B82FF, #6A48E8); color: #fff; font-size: 1.25rem; font-weight: 700;
  box-shadow: 0 8px 20px -8px rgba(106,72,232,.7);
}
.brand-name { font-weight: 800; font-size: 1.15rem; letter-spacing: -0.01em; }

/* Hero */
.hero-title { font-size: clamp(2.2rem, 5vw, 3.6rem); line-height: 1.08; margin: .2rem 0 1rem; font-weight: 800; letter-spacing: -0.03em; }
.hero-sub { font-size: 1.1rem; line-height: 1.7; color: var(--muted); max-width: 34rem; margin-bottom: 1.2rem; }
.chips { display: flex; flex-wrap: wrap; gap: .5rem; margin: .4rem 0 1.4rem; }
.chip { padding: .3rem .85rem; border-radius: 999px; background: var(--lav); border: 1px solid var(--line); color: var(--violet-deep); font-weight: 600; font-size: .95rem; }
.demo {
  background: #fff; border: 1px solid var(--line); border-radius: 28px; padding: 1.6rem 1.7rem;
  box-shadow: 0 30px 60px -30px rgba(90,63,209,.45);
}
.demo-lang { font-size: .85rem; color: var(--muted); font-weight: 600; margin-bottom: .2rem; }
.demo-en { font-size: 1.05rem; color: var(--muted); margin-bottom: 1.1rem; }
.demo-te { font-size: 1.7rem; line-height: 1.5; font-weight: 600; color: var(--violet-deep); margin-bottom: 1.1rem; }
.demo-hi { font-size: 1.15rem; line-height: 1.6; color: var(--ink); }
.demo-rule { border: 0; border-top: 1px dashed var(--line); margin: 0 0 1rem; }

/* Home sections */
.section-title { font-size: 1.5rem; margin: 2.6rem 0 1rem; }
.mode-card { background: #fff; border: 1px solid var(--line); border-radius: 24px; padding: 1.4rem 1.5rem; height: 100%; }
.mode-card.alt { background: linear-gradient(160deg, #F3EFFF 0%, #FFFFFF 100%); }
.mode-card h3 { margin: 0 0 .4rem; font-size: 1.15rem; }
.mode-card p { margin: 0; color: var(--muted); line-height: 1.65; }
.step { display: flex; gap: .9rem; align-items: flex-start; }
.step-n { flex: 0 0 2rem; height: 2rem; border-radius: 50%; background: var(--violet); color: #fff; display: grid; place-items: center; font-weight: 700; }
.step p { margin: .1rem 0 0; color: var(--muted); line-height: 1.6; }
.step b { display: block; color: var(--ink); }
.who { border-top: 2px solid var(--lav-2); padding-top: .9rem; }
.who b { display: block; margin-bottom: .25rem; }
.who span { color: var(--muted); line-height: 1.6; }

/* Workspace cards (containers created with key="card_...") */
div[class*="st-key-card_"] {
  background: #fff; border: 1px solid var(--line); border-radius: 26px; padding: 1.4rem 1.5rem;
  box-shadow: 0 18px 50px -28px rgba(90,63,209,.45); margin-bottom: 1.2rem;
}
.field-label { font-weight: 700; margin-bottom: .35rem; }
.hint { color: var(--muted); font-size: .92rem; }

/* Inputs */
.stTextArea textarea { border-radius: 18px; font-size: 1.05rem; line-height: 1.7; padding: 1rem 1.1rem; }
div[data-baseweb="textarea"], div[data-baseweb="select"] > div { border-radius: 16px !important; border-color: var(--line) !important; background: #fff; }
div[data-baseweb="textarea"]:focus-within { border-color: var(--violet) !important; box-shadow: 0 0 0 3px rgba(123,92,240,.18); }

/* Buttons */
.stButton > button, .stDownloadButton > button {
  border-radius: 999px; font-weight: 600; border: 1px solid var(--line); background: #fff; color: var(--ink); padding: .5rem 1.25rem;
}
.stButton > button:hover { border-color: var(--violet); color: var(--violet-deep); }
button[kind="primary"], button[data-testid="stBaseButton-primary"] {
  background: linear-gradient(135deg, #8F74FF, #6A48E8) !important; color: #fff !important; border: 0 !important;
  box-shadow: 0 10px 24px -12px rgba(106,72,232,.8);
}
button:focus-visible { outline: 3px solid rgba(123,92,240,.45) !important; outline-offset: 2px; }

/* Output */
.out-meta { display: flex; gap: .5rem; flex-wrap: wrap; margin-bottom: .8rem; }
.pill { padding: .2rem .75rem; border-radius: 999px; background: var(--violet); color: #fff; font-weight: 600; font-size: .85rem; }
.pill.soft { background: var(--lav-2); color: var(--violet-deep); }
.out-text { font-size: 1.25rem; line-height: 1.95; color: var(--ink); word-break: break-word; margin-bottom: 1rem; }
.placeholder { text-align: center; color: var(--muted); padding: 2.2rem 1rem; border: 2px dashed var(--line); border-radius: 20px; background: linear-gradient(160deg, #FAF8FF, #fff); }
.status { display: inline-block; padding: .25rem .8rem; border-radius: 999px; font-size: .85rem; font-weight: 600; }
.status.ok { background: #E8F7EE; color: #1E7B45; }
.status.bad { background: #FDECEC; color: #B3261E; }

@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  div[class*="st-key-card_"] { padding: 1rem; border-radius: 20px; }
  .demo { padding: 1.2rem; }
  .demo-te { font-size: 1.4rem; }
  .out-text { font-size: 1.12rem; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
"""

SAMPLES = {
    "Loan approval": "Your loan application has been approved, subject to verification of your documents.",
    "Interest rate": "The interest rate is 9% per annum, compounded quarterly, and must be paid in monthly instalments.",
    "Office visit": "Please carry your Aadhaar card and two passport-size photographs to the office by 15 March.",
}


def _html(block: str) -> str:
    """Flatten HTML so Markdown never mistakes indented lines for code blocks."""
    return " ".join(line.strip() for line in block.strip().splitlines())


def render_brand() -> str:
    return '<div class="brand"><span class="brand-mark">అ</span><span class="brand-name">Bhasha AI</span></div>'


# --------------------------------------------------------------------------- #
# Reusable UI pieces
# --------------------------------------------------------------------------- #


def copy_button(text: str, label: str = "Copy translation") -> None:
    """A real copy-to-clipboard button (runs in the browser, copies visible text only)."""
    payload = json.dumps(text).replace("</", "<\\/")
    components.html(
        f"""
        <style>
          body {{ margin: 0; }}
          button {{ font: 600 14px 'Plus Jakarta Sans', system-ui, sans-serif; color: #241B4B; background: #fff;
                   border: 1px solid #E0D7FA; border-radius: 999px; padding: 9px 20px; cursor: pointer; }}
          button:hover {{ border-color: #7B5CF0; color: #5A3FD1; }}
          button:focus-visible {{ outline: 3px solid rgba(123,92,240,.45); outline-offset: 2px; }}
        </style>
        <button id="b" type="button">{html.escape(label)}</button>
        <script>
          const text = {payload};
          const btn = document.getElementById('b');
          btn.addEventListener('click', async () => {{
            try {{ await navigator.clipboard.writeText(text); }}
            catch (e) {{
              const t = document.createElement('textarea'); t.value = text;
              document.body.appendChild(t); t.select(); document.execCommand('copy'); t.remove();
            }}
            btn.textContent = 'Copied ✓';
            setTimeout(() => btn.textContent = {json.dumps(label)}, 1800);
          }});
        </script>
        """,
        height=52,
    )


@st.cache_data(show_spinner=False, max_entries=64, ttl=3600)
def _speak(text: str, language: str) -> bytes:
    return synthesize_speech(text, language)


def key_banner() -> None:
    if not get_api_key():
        st.error(MISSING_KEY_MESSAGE, icon=":material/key_off:")


# --------------------------------------------------------------------------- #
# Workspace shared by the Translate and Explain pages
# --------------------------------------------------------------------------- #


def _on_audio(page: str) -> None:
    """Runs when a new voice recording is finished (before the page reruns)."""
    audio = st.session_state.get(f"{page}_audio_in")
    if audio is None:
        return
    spoken = st.session_state.get(f"{page}_speech_lang", "English")
    try:
        text = transcribe_audio(audio.getvalue(), getattr(audio, "type", None), spoken)
    except BhashaError as err:
        st.session_state[f"{page}_voice_msg"] = ("error", str(err))
        return
    except Exception:  # never show a stack trace to the user
        logger.exception("Voice input failed")
        st.session_state[f"{page}_voice_msg"] = ("error", "Voice input failed. Please try again or type your text.")
        return
    st.session_state[f"{page}_text"] = text
    st.session_state.pop(f"{page}_result", None)
    st.session_state[f"{page}_voice_msg"] = ("success", f"Heard: {text}")
    if st.session_state.get(f"{page}_auto", True):
        st.session_state[f"{page}_auto_run"] = True


def _clear(page: str) -> None:
    for suffix in ("result", "voice_msg", "auto_run"):
        st.session_state.pop(f"{page}_{suffix}", None)
    st.session_state[f"{page}_text"] = ""


def _load_sample(page: str, text: str) -> None:
    for suffix in ("result", "voice_msg", "auto_run"):
        st.session_state.pop(f"{page}_{suffix}", None)
    st.session_state[f"{page}_text"] = text


def _run(page: str, text: str, language: str, mode: str) -> None:
    st.session_state.pop(f"{page}_result", None)
    verb = "Translating" if mode == MODE_DIRECT else "Explaining"
    try:
        with st.spinner(f"{verb} into {language}…"):
            output = translate_text(text, language, mode)
    except BhashaError as err:
        st.error(str(err), icon=":material/error:")
        return
    except Exception:
        logger.exception("Unexpected failure while translating")
        st.error("Something went wrong on our side. Please try again in a moment.", icon=":material/error:")
        return
    st.session_state[f"{page}_result"] = {"text": output, "language": language, "mode": mode, "audio": None}


def render_workspace(page: str, *, default_mode: str, allow_mode_change: bool) -> None:
    k = lambda name: f"{page}_{name}"  # noqa: E731 - tiny local helper
    st.session_state.setdefault(k("text"), "")
    st.session_state.setdefault(k("lang"), next(iter(LANGUAGES)))  # Telugu
    st.session_state.setdefault(k("mode"), default_mode)
    st.session_state.setdefault(k("auto"), True)

    key_banner()

    # ---- Input card -------------------------------------------------------
    with st.container(key=f"card_input_{page}"):
        head_l, head_r = st.columns([3, 1.4], vertical_alignment="center")
        head_l.markdown('<div class="field-label">English text</div>', unsafe_allow_html=True)
        with head_r.popover("Voice input", icon=":material/mic:", use_container_width=True):
            st.selectbox("I will speak in", SPEECH_LANGUAGES, key=k("speech_lang"))
            st.checkbox("Translate right after I speak", key=k("auto"))
            st.audio_input("Record", key=k("audio_in"), on_change=_on_audio, args=(page,), label_visibility="collapsed")
            st.caption("Allow microphone access when your browser asks.")

        st.text_area(
            "English text",
            key=k("text"),
            height=190,
            max_chars=MAX_INPUT_CHARS,
            placeholder="Type or paste English here, or use Voice input…",
            label_visibility="collapsed",
        )

        voice_msg = st.session_state.get(k("voice_msg"))
        if voice_msg:
            (st.success if voice_msg[0] == "success" else st.error)(voice_msg[1], icon=":material/mic:")

        st.markdown('<div class="hint">Try an example</div>', unsafe_allow_html=True)
        for col, (label, sample) in zip(st.columns(len(SAMPLES)), SAMPLES.items()):
            col.button(label, key=k(f"ex_{label}"), on_click=_load_sample, args=(page, sample), use_container_width=True)

        opt_l, opt_r = st.columns([1, 1.3], gap="large")
        with opt_l:
            language = st.selectbox(
                "Translate into",
                list(LANGUAGES),
                key=k("lang"),
                format_func=lambda n: f"{n} ({LANGUAGES[n].native})",
            )
        with opt_r:
            if allow_mode_change:
                mode = st.radio("Mode", MODES, key=k("mode"), horizontal=True)
            else:
                mode = default_mode
                st.markdown('<div class="field-label">Mode</div>', unsafe_allow_html=True)
                st.markdown(f'<span class="pill soft">{mode}</span>', unsafe_allow_html=True)
            st.markdown(
                '<div class="hint">'
                + (
                    "Translates every sentence faithfully, keeping the meaning."
                    if mode == MODE_DIRECT
                    else "Explains the meaning in easy everyday words, with an example when it helps."
                )
                + "</div>",
                unsafe_allow_html=True,
            )

        act_l, act_r, _ = st.columns([1.3, 1, 2.5])
        go = act_l.button(
            "Translate" if mode == MODE_DIRECT else "Explain simply",
            key=k("go"), type="primary", icon=":material/translate:", use_container_width=True,
        )
        act_r.button("Clear", key=k("clear"), icon=":material/close:", on_click=_clear, args=(page,), use_container_width=True)

    # ---- Processing -------------------------------------------------------
    auto = st.session_state.pop(k("auto_run"), False)
    if go or auto:
        if go:
            st.session_state.pop(k("voice_msg"), None)
        _run(page, st.session_state[k("text")], language, mode)

    # ---- Output card ------------------------------------------------------
    result = st.session_state.get(k("result"))
    with st.container(key=f"card_output_{page}"):
        if not result:
            st.markdown(
                '<div class="placeholder"><b>Your result will appear here</b><br>'
                "Add some English text above, choose a language and press the button.</div>",
                unsafe_allow_html=True,
            )
            return

        lang = LANGUAGES[result["language"]]
        safe = html.escape(result["text"]).replace("$", "&#36;").replace("\n", "<br>")
        st.markdown(
            _html(
                f"""
                <div class="out-meta">
                  <span class="pill">{lang.name} · {lang.native}</span>
                  <span class="pill soft">{html.escape(result['mode'])}</span>
                </div>
                <div class="out-text" lang="{lang.code}">{safe}</div>
                """
            ),
            unsafe_allow_html=True,
        )

        c_copy, c_listen, _ = st.columns([1.3, 1, 2.5], vertical_alignment="center")
        with c_copy:
            copy_button(result["text"])
        listen = c_listen.button("Listen", key=k("listen"), icon=":material/volume_up:", use_container_width=True)
        if listen:
            try:
                with st.spinner("Preparing voice…"):
                    result["audio"] = _speak(result["text"], result["language"])
                st.session_state[k("result")] = result
            except BhashaError as err:
                st.error(str(err), icon=":material/volume_off:")
            except Exception:
                logger.exception("Unexpected text-to-speech failure")
                st.error("Couldn't play the voice right now. Please try again.", icon=":material/volume_off:")
        if result.get("audio"):
            st.audio(result["audio"], format="audio/mp3", autoplay=listen)


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(render_brand(), unsafe_allow_html=True)
        ready = bool(get_api_key())
        st.markdown(
            f'<span class="status {"ok" if ready else "bad"}">'
            f'{"AI connected" if ready else "API key missing"}</span>',
            unsafe_allow_html=True,
        )
        st.caption("Understand every word, in your language.")
        if ready:
            st.caption(f"Model: {get_model_name()}")


def render_home() -> None:
    left, right = st.columns([1.15, 1], gap="large", vertical_alignment="center")
    with left:
        chips = "".join(f'<span class="chip">{l.native}</span>' for l in LANGUAGES.values())
        st.markdown(
            _html(
                f"""
                {render_brand()}
                <h1 class="hero-title">Understand Every Word, In Your Language.</h1>
                <p class="hero-sub">Paste or speak English and get it back in your own language —
                translated word for word, or explained in simple everyday words anyone can follow.</p>
                <div class="chips">{chips}</div>
                """
            ),
            unsafe_allow_html=True,
        )
        b1, b2 = st.columns(2)
        if b1.button("Start translating", type="primary", icon=":material/translate:", use_container_width=True):
            st.switch_page(PAGE_TRANSLATE)
        if b2.button("Explain in simple words", icon=":material/lightbulb:", use_container_width=True):
            st.switch_page(PAGE_EXPLAIN)
    with right:
        st.markdown(
            _html(
                """
                <div class="demo">
                  <div class="demo-lang">English</div>
                  <div class="demo-en">Your application will be reviewed within 7 working days.</div>
                  <hr class="demo-rule">
                  <div class="demo-lang">తెలుగు</div>
                  <div class="demo-te" lang="te">మీ దరఖాస్తును 7 పని దినాల్లో పరిశీలిస్తారు.</div>
                  <div class="demo-lang">हिन्दी</div>
                  <div class="demo-hi" lang="hi">आपके आवेदन की समीक्षा 7 कार्य दिवसों में की जाएगी।</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

    st.markdown('<h2 class="section-title">Two ways to understand</h2>', unsafe_allow_html=True)
    m1, m2 = st.columns(2, gap="large")
    m1.markdown(
        _html(
            """
            <div class="mode-card"><h3>Direct Translation</h3>
            <p>Every sentence is translated faithfully. Names, dates, numbers and amounts stay exactly as written.</p></div>
            """
        ),
        unsafe_allow_html=True,
    )
    m2.markdown(
        _html(
            """
            <div class="mode-card alt"><h3>Simple Explanation</h3>
            <p>Long or formal English is explained in short sentences and easy words, with an everyday example when it helps.</p></div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.markdown('<h2 class="section-title">How it works</h2>', unsafe_allow_html=True)
    steps = [
        ("Type, paste or speak", "Add the English text you want to understand."),
        ("Choose language and mode", "Pick Telugu, Hindi, Tamil, Kannada, Malayalam or Marathi."),
        ("Read it or listen to it", "Copy the result, or press Listen to hear it read aloud."),
    ]
    for col, (i, (title, body)) in zip(st.columns(3, gap="large"), enumerate(steps, 1)):
        col.markdown(
            _html(f'<div class="step"><div class="step-n">{i}</div><div><b>{title}</b><p>{body}</p></div></div>'),
            unsafe_allow_html=True,
        )

    st.markdown('<h2 class="section-title">Made for everyday life</h2>', unsafe_allow_html=True)
    who = [
        ("Students", "Understand textbooks, notices and exam instructions written in English."),
        ("Families and villages", "Make sense of government forms, bank letters and SMS messages."),
        ("Everyone else", "Read medicine labels, job offers and agreements with confidence."),
    ]
    for col, (title, body) in zip(st.columns(3, gap="large"), who):
        col.markdown(_html(f'<div class="who"><b>{title}</b><span>{body}</span></div>'), unsafe_allow_html=True)


def render_translate() -> None:
    st.title("Translate")
    st.caption("Turn English into Telugu, Hindi, Tamil, Kannada, Malayalam or Marathi.")
    render_workspace("translate", default_mode=MODE_DIRECT, allow_mode_change=True)


def render_explain() -> None:
    st.title("Explain in Simple Language")
    st.caption("Confused by a long or formal sentence? Get it explained in easy words.")
    render_workspace("explain", default_mode=MODE_SIMPLE, allow_mode_change=False)


def render_about() -> None:
    st.title("About Project")
    st.markdown(
        """
**Bhasha AI** helps people who are not comfortable with English understand it in their own Indian language.
It is built for students, rural communities and everyday users.

**What it does**
- Translates English into Telugu, Hindi, Tamil, Kannada, Malayalam and Marathi.
- Explains hard English in simple regional-language words, with examples when useful.
- Lets you speak instead of typing, and reads the result aloud.

**How it works**
- Frontend and server: Streamlit and Python.
- AI: Google Gemini through the official Google Gen AI SDK. The API key stays on the server.
- Voice input: your recording is transcribed by Gemini.
- Voice output: Google Text-to-Speech (gTTS).

**Good to know**
- AI can make mistakes. For legal, medical or money decisions, check important details with a qualified person.
- Your text is sent to Google services to be translated and read aloud. Please do not enter passwords, bank PINs or other secrets.
- Text is limited to 5,000 characters at a time. Voice playback needs an internet connection.
        """
    )
    st.markdown("#### Supported languages")
    st.table(
        {
            "Language": list(LANGUAGES),
            "Written as": [l.native for l in LANGUAGES.values()],
            "Voice output": ["Yes"] * len(LANGUAGES),
        }
    )


# --------------------------------------------------------------------------- #
# Navigation
# --------------------------------------------------------------------------- #

PAGE_HOME = st.Page(render_home, title="Home", icon=":material/home:", url_path="home", default=True)
PAGE_TRANSLATE = st.Page(render_translate, title="Translate", icon=":material/translate:", url_path="translate")
PAGE_EXPLAIN = st.Page(render_explain, title="Explain in Simple Language", icon=":material/lightbulb:", url_path="explain")
PAGE_ABOUT = st.Page(render_about, title="About Project", icon=":material/info:", url_path="about")

st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
current = st.navigation([PAGE_HOME, PAGE_TRANSLATE, PAGE_EXPLAIN, PAGE_ABOUT])
render_sidebar()
current.run()
