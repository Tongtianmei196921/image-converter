import glob
import inspect
import os
import queue
import re
import threading
from collections.abc import Generator
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Literal, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from biomni.config import default_config
from biomni.know_how import KnowHowLoader
from biomni.llm import SourceType, get_llm
from biomni.model.retriever import ToolRetriever
from biomni.tool.support_tools import run_python_repl
from biomni.tool.tool_registry import ToolRegistry
from biomni.utils import (
    check_and_download_s3_files,
    clean_message_content,
    convert_markdown_to_pdf,
    create_parsing_error_html,
    find_matching_execution,
    format_execute_tags_in_content,
    format_lists_in_text,
    format_observation_as_terminal,
    function_to_api_schema,
    has_execution_results,
    inject_custom_functions_to_repl,
    parse_tool_calls_from_code,
    parse_tool_calls_with_modules,
    pretty_print,
    read_module2api,
    run_bash_script,
    run_r_code,
    run_with_timeout,
    should_skip_message,
    textify_api_dict,
)

if os.path.exists(".env"):
    load_dotenv(".env", override=False)
    print("Loaded environment variables from .env")


def _gradio_workbench_css() -> str:
    return """
/* ═══════════════════════════════════════════════════════════════
   BIOMNI RESEARCH WORKBENCH  ·  Apple Design Language  ·  v4.0
   ═══════════════════════════════════════════════════════════════ */

:root {
    /* ── Palette ── */
    --p-bg:        #f5f5f7;
    --p-surface:   #ffffff;
    --p-surface-2: #fafafc;
    --p-fill:      rgba(116,116,128,0.10);
    --p-fill-2:    rgba(116,116,128,0.06);
    --p-sep:       rgba(60,60,67,0.12);
    --p-sep-2:     rgba(60,60,67,0.07);

    /* ── Text ── */
    --t-1: #1d1d1f;
    --t-2: #3c3c43;
    --t-3: #6c6c70;
    --t-4: #aeaeb2;

    /* ── System accents ── */
    --a-blue:   #0071e3;
    --a-blue-d: rgba(0,113,227,0.10);
    --a-blue-g: rgba(0,113,227,0.16);
    --a-green:  #30d158;
    --a-grn-d:  rgba(48,209,88,0.14);
    --a-orange: #ff9f0a;
    --a-orng-d: rgba(255,159,10,0.12);
    --a-purple: #bf5af2;
    --a-purp-d: rgba(191,90,242,0.12);

    /* ── Shadows (three-layer, natural) ── */
    --sh-0: 0 0 0 1px rgba(0,0,0,0.04);
    --sh-1: 0 1px 2px rgba(0,0,0,0.03), 0 4px 10px rgba(0,0,0,0.04);
    --sh-2: 0 2px 4px rgba(0,0,0,0.04), 0 8px 22px rgba(0,0,0,0.06);
    --sh-3: 0 4px 8px rgba(0,0,0,0.05), 0 16px 40px rgba(0,0,0,0.08);
    --sh-focus: 0 0 0 4px rgba(0,113,227,0.18);

    /* ── Geometry ── */
    --r-2xl: 22px;
    --r-xl:  16px;
    --r-lg:  12px;
    --r-md:  9px;
    --r-sm:  6px;

    /* ── Motion ── */
    --ease-out:  cubic-bezier(0.25, 0.46, 0.45, 0.94);
    --ease-spring: cubic-bezier(0.34, 1.56, 0.64, 1);

    /* ── Typography ── */
    --f-sans: -apple-system, "SF Pro Display", "SF Pro Text", "Helvetica Neue",
              system-ui, "Segoe UI Variable", "Segoe UI", sans-serif;
    --f-mono: "SF Mono", "Fira Code", "Cascadia Code", "Consolas",
              ui-monospace, monospace;
}

/* ────────────────────────────────────────
   BASE
──────────────────────────────────────── */
.gradio-container {
    font-family: var(--f-sans) !important;
    background: var(--p-bg) !important;
    color: var(--t-1) !important;
    -webkit-font-smoothing: antialiased !important;
    -moz-osx-font-smoothing: grayscale !important;
}

/* Override Gradio CSS vars so inner components inherit correctly */
.gradio-container {
    --body-text-color: var(--t-1) !important;
    --body-text-color-subdued: var(--t-3) !important;
    --color-accent: var(--a-blue) !important;
    --border-color-primary: var(--p-sep) !important;
    --background-fill-primary: var(--p-surface) !important;
    --background-fill-secondary: var(--p-surface-2) !important;
    --block-background-fill: var(--p-surface) !important;
    --block-border-color: var(--p-sep) !important;
    --block-shadow: var(--sh-1) !important;
}

#biomni-shell {
    max-width: 1600px;
    margin: 0 auto;
    padding: 32px 24px 40px;
    display: flex;
    flex-direction: column;
    gap: 0;
}

/* ────────────────────────────────────────
   HEADER
──────────────────────────────────────── */
#biomni-header {
    position: relative;
    background: var(--p-surface);
    border-radius: var(--r-2xl);
    box-shadow: var(--sh-2);
    padding: 40px 48px 36px;
    overflow: hidden;
    isolation: isolate;
}

/* Decorative mesh gradient — upper-right */
#biomni-header::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 360px; height: 360px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(0,113,227,0.07) 0%,
        rgba(191,90,242,0.04) 50%,
        transparent 70%
    );
    pointer-events: none;
    z-index: 0;
}

/* Lower-left accent */
#biomni-header::after {
    content: '';
    position: absolute;
    bottom: -30px; left: 120px;
    width: 200px; height: 200px;
    border-radius: 50%;
    background: radial-gradient(
        circle,
        rgba(48,209,88,0.05) 0%,
        transparent 70%
    );
    pointer-events: none;
    z-index: 0;
}

#biomni-header > * { position: relative; z-index: 1; }

.bm-wordmark {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 18px;
}

.bm-logo-mark {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: linear-gradient(145deg, #0a84ff 0%, #0071e3 100%);
    box-shadow: 0 4px 12px rgba(0,113,227,0.35);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}

.bm-wordmark-text {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--t-2);
    font-family: var(--f-sans);
}

.bm-wordmark-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 999px;
    background: var(--a-grn-d);
    margin-left: 2px;
}

.bm-wordmark-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--a-green);
    animation: live-dot 2.2s ease-in-out infinite;
}

@keyframes live-dot {
    0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(48,209,88,0.4); }
    50%       { opacity: 0.7; box-shadow: 0 0 0 5px rgba(48,209,88,0); }
}

.bm-wordmark-live {
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--a-green);
}

.biomni-header-kicker { display: none; }

.biomni-header-title {
    margin: 0 0 12px;
    font-size: clamp(40px, 4vw, 60px);
    line-height: 1.02;
    letter-spacing: -0.04em;
    font-weight: 700;
    color: var(--t-1);
    font-family: var(--f-sans);
}

.biomni-header-subtitle {
    max-width: 680px;
    margin: 0;
    font-size: 16px;
    line-height: 1.70;
    color: var(--t-3);
    font-weight: 400;
    letter-spacing: -0.01em;
}

/* ── Status strip ── */
#biomni-status-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 6px;
    margin-top: 28px;
    padding-top: 22px;
    border-top: 1px solid var(--p-sep-2);
}

.biomni-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    height: 28px;
    padding: 0 11px;
    border-radius: 999px;
    background: var(--p-fill);
    transition: background 0.18s var(--ease-out);
    cursor: default;
    user-select: none;
}

.biomni-status-pill:hover { background: var(--a-blue-d); }

.biomni-status-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--t-3);
    font-family: var(--f-sans);
}

.biomni-status-sep {
    display: inline-block;
    width: 1px; height: 10px;
    background: var(--p-sep);
}

.biomni-status-value {
    font-size: 12px;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--t-2);
    font-family: var(--f-sans);
}

/* ────────────────────────────────────────
   INFO ROW (hero cards)
──────────────────────────────────────── */
#biomni-hero-row {
    gap: 12px;
    margin-top: 12px;
}

.biomni-panel {
    background: var(--p-surface);
    border-radius: var(--r-2xl);
    box-shadow: var(--sh-2);
    overflow: hidden;
    transition: box-shadow 0.3s var(--ease-out), transform 0.3s var(--ease-out);
}

.biomni-panel:hover {
    box-shadow: var(--sh-3);
    transform: translateY(-2px);
}

.biomni-hero-card { padding: 28px 32px 26px; }

.biomni-panel-kicker {
    display: inline-block;
    margin: 0 0 8px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--a-blue);
    font-family: var(--f-sans);
}

.biomni-hero-title {
    margin: 0 0 8px;
    font-size: 20px;
    line-height: 1.2;
    letter-spacing: -0.03em;
    font-weight: 700;
    color: var(--t-1);
    font-family: var(--f-sans);
}

.biomni-hero-copy {
    margin: 0;
    font-size: 13.5px;
    line-height: 1.65;
    color: var(--t-3);
    letter-spacing: -0.005em;
}

/* ── Quickstart cards ── */
.biomni-quickstart-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 9px;
    margin-top: 18px;
}

.biomni-quickstart-card {
    padding: 15px 17px 15px 20px;
    border-radius: var(--r-xl);
    background: var(--p-surface-2);
    border: 1px solid var(--p-sep-2);
    border-left-width: 3px;
    transition: background 0.2s, box-shadow 0.2s, transform 0.2s var(--ease-spring);
    cursor: default;
}

.biomni-quickstart-card:nth-child(1) { border-left-color: var(--a-orange); }
.biomni-quickstart-card:nth-child(2) { border-left-color: var(--a-green);  }
.biomni-quickstart-card:nth-child(3) { border-left-color: var(--a-purple); }

.biomni-quickstart-card:nth-child(1):hover {
    background: var(--a-orng-d);
    box-shadow: 0 4px 14px rgba(255,159,10,0.10);
    transform: translateY(-2px);
}

.biomni-quickstart-card:nth-child(2):hover {
    background: var(--a-grn-d);
    box-shadow: 0 4px 14px rgba(48,209,88,0.10);
    transform: translateY(-2px);
}

.biomni-quickstart-card:nth-child(3):hover {
    background: var(--a-purp-d);
    box-shadow: 0 4px 14px rgba(191,90,242,0.10);
    transform: translateY(-2px);
}

.biomni-quickstart-label {
    margin: 0 0 5px;
    font-size: 13px;
    font-weight: 600;
    color: var(--t-2);
    letter-spacing: -0.01em;
    font-family: var(--f-sans);
}

.biomni-quickstart-text {
    margin: 0;
    font-size: 12px;
    line-height: 1.55;
    color: var(--t-3);
    letter-spacing: -0.005em;
}

/* ── Run-guide list ── */
.biomni-runlist {
    margin: 16px 0 0;
    padding: 0;
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 13px;
}

.biomni-runlist li {
    display: flex;
    gap: 12px;
    align-items: flex-start;
    font-size: 13.5px;
    line-height: 1.55;
    color: var(--t-3);
    letter-spacing: -0.005em;
}

.biomni-runlist strong {
    color: var(--t-2);
    font-weight: 600;
}

.biomni-runlist-marker {
    width: 6px; height: 6px;
    margin-top: 6px;
    border-radius: 50%;
    background: var(--a-blue);
    flex-shrink: 0;
    box-shadow: 0 0 0 3px var(--a-blue-d);
}

/* ────────────────────────────────────────
   WORKBENCH (chat panels)
──────────────────────────────────────── */
#biomni-workbench {
    gap: 12px;
    margin-top: 12px;
}

.biomni-panel-head {
    padding: 18px 24px 16px;
    border-bottom: 1px solid var(--p-sep-2);
    background: var(--p-surface);
}

.biomni-panel-title {
    margin: 0;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: var(--t-1);
    font-family: var(--f-sans);
}

.biomni-panel-copy {
    margin: 4px 0 0;
    font-size: 12.5px;
    line-height: 1.55;
    color: var(--t-3);
    letter-spacing: -0.005em;
}

/* Chat bubbles */
#biomni-main-chat,
#biomni-trace-chat {
    background: transparent !important;
}

#biomni-main-chat .bubble-wrap,
#biomni-trace-chat .bubble-wrap {
    padding: 16px 18px !important;
}

#biomni-main-chat .message,
#biomni-trace-chat .message {
    border-radius: var(--r-xl) !important;
    border: none !important;
    box-shadow: var(--sh-1) !important;
    font-family: var(--f-sans) !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    letter-spacing: -0.008em !important;
    transition: box-shadow 0.2s !important;
}

#biomni-main-chat .message.user,
#biomni-trace-chat .message.user {
    background: linear-gradient(150deg, #e5f0ff 0%, #dae8ff 100%) !important;
    color: var(--t-1) !important;
}

#biomni-main-chat .message.bot {
    background: var(--p-surface) !important;
    color: var(--t-1) !important;
}

#biomni-trace-chat .message.bot {
    background: var(--p-surface) !important;
    border-left: 2px solid rgba(0,113,227,0.22) !important;
    border-radius: 0 var(--r-xl) var(--r-xl) 0 !important;
    color: var(--t-2) !important;
}

/* ────────────────────────────────────────
   COMPOSER
──────────────────────────────────────── */
#biomni-composer {
    margin-top: 12px;
    padding: 22px 26px 16px;
}

.biomni-composer-meta {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 16px;
    margin-bottom: 14px;
}

.biomni-composer-title {
    margin: 0;
    font-size: 14px;
    font-weight: 600;
    color: var(--t-1);
    letter-spacing: -0.02em;
    font-family: var(--f-sans);
}

.biomni-composer-copy {
    margin: 3px 0 0;
    font-size: 12.5px;
    line-height: 1.5;
    color: var(--t-3);
    letter-spacing: -0.005em;
}

.biomni-composer-hint {
    flex-shrink: 0;
    padding: 5px 12px;
    border-radius: 999px;
    background: var(--a-blue-d);
    color: var(--a-blue);
    font-size: 11px;
    font-weight: 600;
    white-space: nowrap;
    font-family: var(--f-sans);
    letter-spacing: -0.01em;
}

/* Model routing */
#biomni-llm-routing {
    margin-bottom: 12px;
    border-radius: var(--r-xl) !important;
    border: 1px solid var(--p-sep) !important;
    background: var(--p-surface-2) !important;
    overflow: hidden;
    box-shadow: none !important;
}

.biomni-llm-note {
    font-size: 13px;
    line-height: 1.55;
    color: var(--t-3);
    margin: 0 0 12px;
}

#biomni-llm-feedback {
    margin: 6px 0 0;
    font-size: 12.5px;
}

/* Quick actions */
#biomni-quick-actions {
    gap: 7px;
    margin-bottom: 10px;
    flex-wrap: wrap;
    position: relative;
    z-index: 6;
    pointer-events: auto !important;
}

#biomni-quick-actions > * {
    position: relative;
    z-index: 6;
    pointer-events: auto !important;
}

.biomni-quick-btn {
    height: 33px !important;
    padding: 0 15px !important;
    border-radius: 999px !important;
    border: 1px solid var(--p-sep) !important;
    background: var(--p-surface) !important;
    color: var(--t-2) !important;
    font-family: var(--f-sans) !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    letter-spacing: -0.01em !important;
    box-shadow: var(--sh-0) !important;
    transition: background 0.18s, border-color 0.18s,
                color 0.18s, transform 0.18s var(--ease-spring),
                box-shadow 0.18s !important;
    white-space: nowrap !important;
    position: relative;
    z-index: 6;
    pointer-events: auto !important;
}

.biomni-quick-btn:hover {
    background: var(--a-blue-d) !important;
    border-color: rgba(0,113,227,0.22) !important;
    color: var(--a-blue) !important;
    transform: translateY(-1.5px) !important;
    box-shadow: 0 4px 12px rgba(0,113,227,0.12) !important;
}

.biomni-quick-btn:active {
    transform: scale(0.96) !important;
}

/* Prompt input */
#biomni-prompt {
    border-radius: var(--r-xl) !important;
    border: 1px solid var(--p-sep) !important;
    background: var(--p-surface) !important;
    box-shadow: var(--sh-1) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    position: relative;
    z-index: 1;
}

#biomni-prompt:focus-within {
    border-color: var(--a-blue) !important;
    box-shadow: var(--sh-focus), var(--sh-1) !important;
}

#biomni-prompt textarea {
    font-size: 14px !important;
    color: var(--t-1) !important;
    font-family: var(--f-sans) !important;
    line-height: 1.65 !important;
    letter-spacing: -0.01em !important;
    background: transparent !important;
}

#biomni-prompt textarea::placeholder {
    color: var(--t-4) !important;
}

/* ────────────────────────────────────────
   GLOBAL GRADIO OVERRIDES
──────────────────────────────────────── */
#biomni-shell * { font-family: var(--f-sans) !important; }

#biomni-shell button {
    border-radius: 999px !important;
    letter-spacing: -0.01em !important;
    transition: all 0.18s var(--ease-out) !important;
}

#biomni-shell .prose { color: var(--t-1) !important; }

#biomni-shell code,
#biomni-shell pre {
    font-family: var(--f-mono) !important;
    font-size: 12.5px !important;
    border-radius: var(--r-lg) !important;
    background: var(--p-surface-2) !important;
    border: 1px solid var(--p-sep-2) !important;
    line-height: 1.6 !important;
    color: var(--t-2) !important;
}

#biomni-shell input:not([type=submit]):not([type=button]),
#biomni-shell select {
    font-family: var(--f-sans) !important;
    font-size: 13.5px !important;
    color: var(--t-1) !important;
    border-radius: var(--r-lg) !important;
    border-color: var(--p-sep) !important;
    background: var(--p-surface) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

#biomni-shell input:focus,
#biomni-shell select:focus {
    border-color: var(--a-blue) !important;
    box-shadow: 0 0 0 3px var(--a-blue-d) !important;
    outline: none !important;
}

#biomni-shell label span,
#biomni-shell .label-wrap span {
    font-size: 11.5px !important;
    font-weight: 500 !important;
    color: var(--t-3) !important;
    letter-spacing: 0.01em !important;
}

/* Accordion */
#biomni-shell .accordion,
#biomni-shell details {
    border-radius: var(--r-xl) !important;
    border-color: var(--p-sep) !important;
    background: var(--p-surface-2) !important;
}

/* Language selector */
#biomni-language-selector {
    max-width: 200px;
    margin-top: 10px;
}

/* Verify */
#biomni-verify {
    max-width: 440px;
    margin: 80px auto 0;
    padding: 36px 36px 28px;
}

/* Thin scrollbars */
#biomni-shell ::-webkit-scrollbar { width: 5px; height: 5px; }
#biomni-shell ::-webkit-scrollbar-track { background: transparent; }
#biomni-shell ::-webkit-scrollbar-thumb {
    background: rgba(0,0,0,0.10);
    border-radius: 999px;
}
#biomni-shell ::-webkit-scrollbar-thumb:hover {
    background: rgba(0,0,0,0.20);
}

/* ────────────────────────────────────────
   KILL: info cards are patronising clutter
──────────────────────────────────────── */
#biomni-hero-row {
    display: none !important;
    margin: 0 !important;
    padding: 0 !important;
    height: 0 !important;
}

/* ────────────────────────────────────────
   HEADER  —  tighter, language in corner
──────────────────────────────────────── */
#biomni-header {
    padding: 28px 36px 22px !important;
}

/* Language selector floats to top-right of header */
#biomni-language-selector {
    position: absolute !important;
    top: 24px !important;
    right: 32px !important;
    max-width: 180px !important;
    margin: 0 !important;
    z-index: 10;
}

#biomni-language-selector label { display: none !important; }

#biomni-language-selector .wrap,
#biomni-language-selector select,
#biomni-language-selector input {
    height: 30px !important;
    min-height: 30px !important;
    font-size: 12px !important;
    border-radius: 999px !important;
    padding: 0 12px !important;
    border-color: var(--p-sep) !important;
    background: var(--p-fill) !important;
    color: var(--t-3) !important;
    box-shadow: none !important;
}

/* Header subtitle — one tight line, nothing more */
.biomni-header-subtitle {
    font-size: 14px !important;
    line-height: 1.5 !important;
    color: var(--t-3) !important;
    max-width: 600px;
}

/* Status pills — slimmer */
#biomni-status-row {
    margin-top: 18px !important;
    padding-top: 16px !important;
    gap: 5px !important;
}

.biomni-status-pill {
    height: 24px !important;
    padding: 0 9px !important;
    gap: 5px !important;
}

.biomni-status-label { font-size: 9.5px !important; }
.biomni-status-value { font-size: 11.5px !important; }
.biomni-status-sep   { height: 8px !important; }

/* ────────────────────────────────────────
   PANEL HEADERS  —  title only, no copy
──────────────────────────────────────── */
.biomni-panel-head {
    padding: 14px 20px 12px !important;
}

.biomni-panel-copy { display: none !important; }

.biomni-panel-kicker {
    font-size: 9.5px !important;
    margin-bottom: 2px !important;
}

.biomni-panel-title { font-size: 15px !important; }

/* ────────────────────────────────────────
   COMPOSER  —  strip the meta copy
──────────────────────────────────────── */
.biomni-composer-meta { display: none !important; }

#biomni-composer {
    padding: 14px 20px 12px !important;
    margin-top: 10px !important;
}

/* ────────────────────────────────────────
   SHELL  —  tighter vertical rhythm
──────────────────────────────────────── */
#biomni-shell {
    padding: 24px 20px 32px !important;
    gap: 0 !important;
}

#biomni-workbench { margin-top: 10px !important; }

/* ────────────────────────────────────────
   RESPONSIVE
──────────────────────────────────────── */
@media (max-width: 1200px) {
    #biomni-workbench { gap: 10px; }
}

@media (max-width: 900px) {
    #biomni-shell { padding: 14px 10px 20px !important; }
    #biomni-header { padding: 24px 18px 18px !important; }
    .biomni-header-title { font-size: clamp(28px, 8vw, 42px); }
    #biomni-workbench { flex-direction: column; }
    #biomni-main-chat, #biomni-trace-chat { min-height: 340px; }
    #biomni-composer { padding: 14px 14px 10px !important; }
    #biomni-language-selector { position: static !important; max-width: 160px !important; margin-top: 10px !important; }
    #biomni-header::before, #biomni-header::after { display: none; }
}
    """


def _gradio_status_pill_html(label: str, value: str) -> str:
    return (
        "<div class='biomni-status-pill'>"
        f"<span class='biomni-status-label'>{escape(label)}</span>"
        "<span class='biomni-status-sep'></span>"
        f"<span class='biomni-status-value'>{escape(value)}</span>"
        "</div>"
    )


_GRADIO_LOCALE_COPY: dict[str, dict[str, Any]] = {
    "en": {
        "language_label": "Language",
        "language_choices": [("Chinese (Mixed Terms)", "zh"), ("English", "en")],
        "header_kicker": "Biomni Interface",
        "header_title": "Research Workbench",
        "header_subtitle": "Run biomedical workflows with live execution traces, file-aware prompts, and a cleaner separation between final answers and operational detail.",
        "composer_title": "New Research Run",
        "composer_copy": "Ask a question, attach a dataset or PDF, and press Enter to launch a new Biomni execution.",
        "composer_hint": "Shift+Enter for a new line",
        "quickstart_kicker": "Quick Starts",
        "quickstart_title": "Start from a proven research track",
        "quickstart_copy": "Use the sample prompts below to enter the workbench with tasks that already map well to Biomni's tools.",
        "quickstart_drug_label": "Drug Label Review",
        "quickstart_drug_copy": "Pull structured FDA information, identify active ingredients, and summarize regulatory details.",
        "quickstart_literature": "Literature Triage",
        "quickstart_literature_copy": "Investigate a biomedical question, retrieve relevant evidence, and return a cleaner answer with traceability.",
        "quickstart_single_cell": "Single-Cell Readout",
        "quickstart_single_cell_copy": "Point Biomni at a dataset and ask for annotation, interpretation, or hypothesis generation.",
        "run_guide_kicker": "Run Discipline",
        "run_guide_title": "Use the split view intentionally",
        "run_guide_copy": "Keep the left pane for the narrative you want to keep, and use the right pane to inspect how the result was produced.",
        "run_guide_console_label": "Console",
        "run_guide_console_copy": "Read the polished answer, upload files, and drive the next step.",
        "run_guide_trace_label": "Trace",
        "run_guide_trace_copy": "Watch tool selection, code execution, observations, and generated artifacts.",
        "run_guide_data_lake_label": "Data Lake",
        "run_guide_data_lake_copy": "This local setup stays on-demand unless a task explicitly needs the larger asset bundle.",
        "model_routing_title": "Model Routing",
        "model_routing_copy": "Switch the active model without editing code. The change applies to new runs and uses OpenAI-compatible routing.",
        "provider_preset_label": "Provider Preset",
        "model_label": "Model",
        "model_info": "Choose a suggested model or type a custom model id.",
        "base_url_label": "Base URL",
        "api_key_label": "API Key",
        "api_key_placeholder": "Leave blank to keep the current key",
        "apply_model_routing": "Apply Model Routing",
        "status_provider": "Provider",
        "status_model": "Model",
        "status_mode": "Mode",
        "status_data_lake": "Data Lake",
        "status_retriever": "Retriever",
        "status_status": "Status",
        "status_academic": "Academic",
        "status_commercial": "Commercial",
        "status_data_lake_loaded": "Loaded",
        "status_data_lake_on_demand": "On demand",
        "status_enabled": "Enabled",
        "status_disabled": "Disabled",
        "status_ready": "Ready",
        "panel_primary_kicker": "Primary",
        "panel_primary_title": "Research Console",
        "panel_primary_copy": "Review the conversation and keep the final answer clean, prominent, and easy to scan.",
        "panel_operations_kicker": "Operations",
        "panel_operations_title": "Execution Trace",
        "panel_operations_copy": "Inspect retrieval, reasoning, code execution, observations, and generated files as the run unfolds.",
        "quick_action_drug": "Drug Label Review",
        "quick_action_literature": "Literature Triage",
        "quick_action_single_cell": "Single-Cell Readout",
        "quick_action_clear": "Clear",
        "prompt_placeholder": "Ask Biomni to investigate a paper, analyze a dataset, validate a drug, or plan an experiment...",
        "verification_copy": "Enter the access code to open the Biomni research workspace.",
        "verification_label": "Access Code",
        "verification_button": "Verify Access",
        "apply_success": "Applied {provider} routing with model `{model}`. New runs will use `{base_url}`.",
        "apply_failure": "Failed to apply model routing: {error}",
        "preset_drug_prompt": "Review the FDA label for ibuprofen and summarize the active ingredient, dosage form, route, product type, and any safety notes that matter for a quick screening.",
        "preset_literature_prompt": "Investigate the latest evidence on CRISPR screening approaches for drug resistance in cancer and return a concise literature triage with key papers, methods, and open questions.",
        "preset_single_cell_prompt": "Inspect the uploaded single-cell dataset, identify the major cell states, and suggest the most important biological readouts or follow-up analyses.",
    },
    "zh": {
        "language_label": "界面语言",
        "language_choices": [("中文（专业混排）", "zh"), ("English", "en")],
        "header_kicker": "Biomni 界面",
        "header_title": "研究工作台",
        "header_subtitle": "发起生物医学工作流，实时查看执行 Trace、文件感知提示，以及最终结果与过程细节的清晰分层。",
        "composer_title": "新建研究任务",
        "composer_copy": "输入问题，附加数据集或 PDF，然后按 Enter 启动一次新的 Biomni 执行。",
        "composer_hint": "Shift+Enter 换行",
        "quickstart_kicker": "快速开始",
        "quickstart_title": "从成熟研究路径直接发起",
        "quickstart_copy": "下面这些示例任务已经和 Biomni 的能力路径较好匹配，适合直接进入工作台使用。",
        "quickstart_drug_label": "药品标签审阅",
        "quickstart_drug_copy": "提取 FDA 结构化信息，识别活性成分，并快速汇总关键监管细节。",
        "quickstart_literature": "文献分诊",
        "quickstart_literature_copy": "围绕一个生物医学问题检索证据，并返回更清晰、可追溯的研究结论。",
        "quickstart_single_cell": "单细胞解读",
        "quickstart_single_cell_copy": "将单细胞数据集交给 Biomni，完成注释、解读或后续假设生成。",
        "run_guide_kicker": "运行准则",
        "run_guide_title": "有意识地使用双栏视图",
        "run_guide_copy": "左侧保留你真正想留下的研究叙事，右侧则用来检查结果是如何被生成出来的。",
        "run_guide_console_label": "结果 Console",
        "run_guide_console_copy": "阅读整理后的答案、上传文件，并推动下一步研究动作。",
        "run_guide_trace_label": "执行 Trace",
        "run_guide_trace_copy": "查看工具选择、代码执行、观察结果和生成产物的全过程。",
        "run_guide_data_lake_label": "Data Lake",
        "run_guide_data_lake_copy": "当前本地环境默认按需加载，只有在任务明确需要时才会拉起更大的数据资产。",
        "model_routing_title": "模型路由",
        "model_routing_copy": "无需改代码即可切换当前活跃模型。新的任务会按这里的 OpenAI-compatible 路由配置执行。",
        "provider_preset_label": "供应商预设",
        "model_label": "Model",
        "model_info": "可直接选择推荐模型，或手动输入自定义 model id。",
        "base_url_label": "Base URL",
        "api_key_label": "API Key",
        "api_key_placeholder": "留空则继续使用当前已加载的 key",
        "apply_model_routing": "应用模型配置",
        "status_provider": "供应商",
        "status_model": "模型",
        "status_mode": "模式",
        "status_data_lake": "Data Lake",
        "status_retriever": "检索器",
        "status_status": "状态",
        "status_academic": "学术版",
        "status_commercial": "商业版",
        "status_data_lake_loaded": "已加载",
        "status_data_lake_on_demand": "按需加载",
        "status_enabled": "已启用",
        "status_disabled": "已关闭",
        "status_ready": "就绪",
        "panel_primary_kicker": "主视图",
        "panel_primary_title": "研究 Console",
        "panel_primary_copy": "在这里阅读整理后的答案，并保持最终结论清晰、醒目、易于扫描。",
        "panel_operations_kicker": "运行区",
        "panel_operations_title": "执行 Trace",
        "panel_operations_copy": "在这里查看检索、推理、代码执行、观察结果与生成文件的完整过程。",
        "quick_action_drug": "药品标签审阅",
        "quick_action_literature": "文献分诊",
        "quick_action_single_cell": "单细胞解读",
        "quick_action_clear": "清空",
        "prompt_placeholder": "让 Biomni 检索论文、分析数据集、验证药物，或规划下一步实验……",
        "verification_copy": "输入访问码以打开 Biomni 研究工作台。",
        "verification_label": "访问码",
        "verification_button": "验证访问",
        "apply_success": "已切换到 {provider} 路由，当前模型为 `{model}`。新的任务会使用 `{base_url}`。",
        "apply_failure": "应用模型配置失败：{error}",
        "preset_drug_prompt": "请审阅布洛芬的 FDA 说明书，概括活性成分、剂型、给药途径、产品类型，以及值得快速筛查关注的安全信息。",
        "preset_literature_prompt": "请调研癌症耐药相关 CRISPR 筛选方法的最新证据，并输出一份简明的文献分诊，包含关键论文、方法路线与未解决问题。",
        "preset_single_cell_prompt": "请检查已上传的单细胞数据集，识别主要细胞状态，并建议最值得优先解读的生物学读数或后续分析方向。",
    },
}


def _gradio_locale_copy(locale: str = "en") -> dict[str, Any]:
    return _GRADIO_LOCALE_COPY.get(locale, _GRADIO_LOCALE_COPY["en"])


def _gradio_header_html(locale: str = "en") -> str:
    copy = _gradio_locale_copy(locale)
    return f"""
    <div class="bm-wordmark">
      <div class="bm-logo-mark">&#x1F9EC;</div>
      <span class="bm-wordmark-text">Biomni</span>
      <span class="bm-wordmark-badge">
        <span class="bm-wordmark-dot"></span>
        <span class="bm-wordmark-live">Live</span>
      </span>
    </div>
    <h1 class="biomni-header-title">{escape(copy["header_title"])}</h1>
    <p class="biomni-header-subtitle">{escape(copy["header_subtitle"])}</p>
    """


def _gradio_panel_header_html(kicker: str, title: str, copy: str) -> str:  # noqa: ARG001
    return (
        "<div class='biomni-panel-head'>"
        f"<p class='biomni-panel-kicker'>{escape(kicker)}</p>"
        f"<h2 class='biomni-panel-title'>{escape(title)}</h2>"
        "</div>"
    )


def _gradio_composer_html(locale: str = "en") -> str:
    copy = _gradio_locale_copy(locale)
    return f"<span class='biomni-composer-hint' style='display:inline-block;margin-bottom:8px'>{escape(copy['composer_hint'])}</span>"


def _gradio_quickstart_html(locale: str = "en") -> str:
    copy = _gradio_locale_copy(locale)
    return f"""
    <div class="biomni-hero-card">
      <p class="biomni-panel-kicker">{escape(copy["quickstart_kicker"])}</p>
      <h3 class="biomni-hero-title">{escape(copy["quickstart_title"])}</h3>
      <p class="biomni-hero-copy">{escape(copy["quickstart_copy"])}</p>
      <div class="biomni-quickstart-grid">
        <div class="biomni-quickstart-card">
          <p class="biomni-quickstart-label">{escape(copy["quickstart_drug_label"])}</p>
          <p class="biomni-quickstart-text">{escape(copy["quickstart_drug_copy"])}</p>
        </div>
        <div class="biomni-quickstart-card">
          <p class="biomni-quickstart-label">{escape(copy["quickstart_literature"])}</p>
          <p class="biomni-quickstart-text">{escape(copy["quickstart_literature_copy"])}</p>
        </div>
        <div class="biomni-quickstart-card">
          <p class="biomni-quickstart-label">{escape(copy["quickstart_single_cell"])}</p>
          <p class="biomni-quickstart-text">{escape(copy["quickstart_single_cell_copy"])}</p>
        </div>
      </div>
    </div>
    """


def _gradio_run_guide_html(locale: str = "en") -> str:
    copy = _gradio_locale_copy(locale)
    return f"""
    <div class="biomni-hero-card">
      <p class="biomni-panel-kicker">{escape(copy["run_guide_kicker"])}</p>
      <h3 class="biomni-hero-title">{escape(copy["run_guide_title"])}</h3>
      <p class="biomni-hero-copy">{escape(copy["run_guide_copy"])}</p>
      <ul class="biomni-runlist">
        <li>
          <span class="biomni-runlist-marker"></span>
          <span><strong>{escape(copy["run_guide_console_label"])}</strong>&ensp;{escape(copy["run_guide_console_copy"])}</span>
        </li>
        <li>
          <span class="biomni-runlist-marker"></span>
          <span><strong>{escape(copy["run_guide_trace_label"])}</strong>&ensp;{escape(copy["run_guide_trace_copy"])}</span>
        </li>
        <li>
          <span class="biomni-runlist-marker"></span>
          <span><strong>{escape(copy["run_guide_data_lake_label"])}</strong>&ensp;{escape(copy["run_guide_data_lake_copy"])}</span>
        </li>
      </ul>
    </div>
    """


_GRADIO_LLM_PROVIDER_PRESETS: dict[str, dict[str, Any]] = {
    "LongCat": {
        "source": "Custom",
        "base_url": "https://api.longcat.chat/openai",
        "default_model": "LongCat-Flash-Thinking-2601",
        "aliases": ["longcat"],
        "models": [
            "LongCat-Flash-Thinking-2601",
        ],
    },
    "DeepSeek": {
        "source": "Custom",
        "base_url": "https://api.deepseek.com",
        "default_model": "deepseek-v4-flash",
        "aliases": ["deepseek"],
        "models": [
            "deepseek-v4-flash",
            "deepseek-v4-pro",
            "deepseek-chat",
            "deepseek-reasoner",
        ],
    },
    "Zhipu": {
        "source": "Custom",
        "base_url": "https://open.bigmodel.cn/api/paas/v4/",
        "default_model": "glm-5.1",
        "aliases": ["zhipu", "glm"],
        "models": [
            "glm-5.1",
            "glm-4.7",
            "glm-4.5",
            "glm-4.5-air",
        ],
    },
    "Custom": {
        "source": "Custom",
        "base_url": "",
        "default_model": "",
        "models": [],
    },
}


def _gradio_llm_provider_presets() -> dict[str, dict[str, Any]]:
    return {
        name: {
            **profile,
            "aliases": list(profile.get("aliases", [])),
            "models": list(profile["models"]),
        }
        for name, profile in _GRADIO_LLM_PROVIDER_PRESETS.items()
    }


def _gradio_guess_llm_provider(model_name: str) -> str:
    normalized_model = (model_name or "").strip().lower()
    if not normalized_model:
        return "Custom"

    for provider_name, profile in _GRADIO_LLM_PROVIDER_PRESETS.items():
        aliases = [provider_name.lower(), *[str(alias).lower() for alias in profile.get("aliases", [])]]
        if normalized_model in aliases:
            return provider_name

        if any(normalized_model == candidate.lower() for candidate in profile["models"]):
            return provider_name

    if normalized_model.startswith("deepseek-"):
        return "DeepSeek"
    if normalized_model.startswith("glm-"):
        return "Zhipu"
    if "longcat" in normalized_model:
        return "LongCat"
    return "Custom"


def _gradio_resolve_runtime_llm_settings(
    provider_name: str,
    model_name: str = "",
    base_url: str = "",
) -> dict[str, str]:
    presets = _gradio_llm_provider_presets()
    selected_provider = provider_name if provider_name in presets else "Custom"
    normalized_model = (model_name or "").strip()
    normalized_base_url = (base_url or "").strip()

    if selected_provider != "Custom":
        profile = presets[selected_provider]
        if not normalized_model:
            normalized_model = str(profile["default_model"])
        if not normalized_base_url:
            normalized_base_url = str(profile["base_url"])
        return {
            "provider": selected_provider,
            "model": normalized_model,
            "base_url": normalized_base_url,
        }

    if not normalized_base_url:
        guessed_provider = _gradio_guess_llm_provider(normalized_model)
        if guessed_provider != "Custom":
            profile = presets[guessed_provider]
            alias_pool = {
                guessed_provider.lower(),
                *[str(alias).lower() for alias in profile.get("aliases", [])],
            }
            selected_provider = guessed_provider
            normalized_base_url = str(profile["base_url"])
            if not normalized_model or normalized_model.lower() in alias_pool:
                normalized_model = str(profile["default_model"])

    return {
        "provider": selected_provider,
        "model": normalized_model,
        "base_url": normalized_base_url,
    }


def _gradio_match_llm_provider(model_name: str, base_url: str | None) -> str:
    normalized_model = (model_name or "").strip()
    normalized_url = (base_url or "").strip().rstrip("/")
    for provider_name, profile in _GRADIO_LLM_PROVIDER_PRESETS.items():
        preset_url = str(profile["base_url"]).strip().rstrip("/")
        if preset_url and normalized_url == preset_url and normalized_model in profile["models"]:
            return provider_name
    guessed_provider = _gradio_guess_llm_provider(normalized_model)
    if guessed_provider != "Custom":
        guessed_profile = _GRADIO_LLM_PROVIDER_PRESETS[guessed_provider]
        guessed_url = str(guessed_profile["base_url"]).strip().rstrip("/")
        if guessed_url and normalized_url == guessed_url:
            return guessed_provider
    return "Custom"


def _gradio_llm_provider_defaults(
    provider_name: str,
    current_model: str = "",
    current_base_url: str = "",
) -> dict[str, str]:
    presets = _gradio_llm_provider_presets()
    resolved = _gradio_resolve_runtime_llm_settings(
        provider_name,
        model_name=current_model,
        base_url=current_base_url,
    )
    selected_provider = resolved["provider"]
    profile = presets[selected_provider]

    current_model = resolved["model"]
    current_base_url = resolved["base_url"]

    if selected_provider == "Custom":
        model = current_model
        base_url = current_base_url
    else:
        preset_base_url = str(profile["base_url"])
        model = current_model if current_model and current_base_url.rstrip("/") == preset_base_url.rstrip("/") else str(
            profile["default_model"]
        )
        base_url = preset_base_url

    return {
        "provider": selected_provider,
        "source": str(profile["source"]),
        "model": model,
        "base_url": base_url,
    }


def _gradio_llm_provider_form_state(
    provider_name: str,
    current_model: str = "",
    current_base_url: str = "",
) -> dict[str, Any]:
    defaults = _gradio_llm_provider_defaults(
        provider_name,
        current_model=current_model,
        current_base_url=current_base_url,
    )
    presets = _gradio_llm_provider_presets()
    profile = presets[defaults["provider"]]
    return {
        **defaults,
        "model_choices": list(profile["models"]),
    }


def _gradio_prompt_presets() -> dict[str, str]:
    return _gradio_prompt_presets_for_locale("en")


def _gradio_prompt_presets_for_locale(locale: str = "en") -> dict[str, str]:
    copy = _gradio_locale_copy(locale)
    return {
        "Drug Label Review": copy["preset_drug_prompt"],
        "Literature Triage": copy["preset_literature_prompt"],
        "Single-Cell Readout": copy["preset_single_cell_prompt"],
    }


def _gradio_prompt_payload(preset_name: str, locale: str = "en") -> dict[str, Any]:
    presets = _gradio_prompt_presets_for_locale(locale)
    return {"text": presets[preset_name], "files": []}


def _gradio_empty_prompt_payload() -> dict[str, Any]:
    return {"text": "", "files": []}


def _gradio_working_message() -> str:
    return "Biomni is working through the request."


def _gradio_timeout_message(timeout_seconds: int) -> str:
    return (
        f"Biomni did not report progress for {timeout_seconds} seconds. "
        "The run was stopped so you can retry with a fresh task."
    )


def _gradio_failure_message(error: Exception) -> str:
    return f"Biomni hit an execution error: {error}"


def _gradio_stream_timeout_seconds(timeout_seconds: int | None) -> int:
    if timeout_seconds is None:
        return 120
    return max(30, min(int(timeout_seconds), 120))


class AgentState(TypedDict):
    messages: list[BaseMessage]
    next_step: str | None


class A1:
    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        source: SourceType | None = None,
        use_tool_retriever: bool | None = None,
        timeout_seconds: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        commercial_mode: bool | None = None,
        expected_data_lake_files: list | None = None,
    ):
        """Initialize the biomni agent.

        Args:
            path: Path to the data
            llm: LLM to use for the agent
            source (str): Source provider: "OpenAI", "AzureOpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", or "Custom"
            use_tool_retriever: If True, use a tool retriever
            timeout_seconds: Timeout for code execution in seconds
            base_url: Base URL for custom model serving (e.g., "http://localhost:8000/v1")
            api_key: API key for the custom LLM
            commercial_mode: If True, excludes datasets that require commercial licenses or are non-commercial only

        """
        # Use default_config values for unspecified parameters
        if path is None:
            path = default_config.path
        if llm is None:
            llm = default_config.llm
        if source is None:
            source = default_config.source
        if use_tool_retriever is None:
            use_tool_retriever = default_config.use_tool_retriever
        if timeout_seconds is None:
            timeout_seconds = default_config.timeout_seconds
        if base_url is None:
            base_url = default_config.base_url
        if api_key is None:
            api_key = default_config.api_key if default_config.api_key else "EMPTY"
        if commercial_mode is None:
            commercial_mode = default_config.commercial_mode

        # Import appropriate env_desc based on commercial_mode
        if commercial_mode:
            from biomni.env_desc_cm import data_lake_dict, library_content_dict

            print("🏢 Commercial mode: Using commercial-licensed datasets only")
        else:
            from biomni.env_desc import data_lake_dict, library_content_dict

            print("🎓 Academic mode: Using all datasets (including non-commercial)")

        # Store as instance attributes for later use
        self.data_lake_dict = data_lake_dict
        self.library_content_dict = library_content_dict
        self.commercial_mode = commercial_mode

        # Display configuration in a nice, readable format
        print("\n" + "=" * 50)
        print("🔧 BIOMNI CONFIGURATION")
        print("=" * 50)

        # Get the actual LLM values that will be used by the agent
        agent_llm = llm if llm is not None else default_config.llm
        agent_source = source if source is not None else default_config.source

        # Show default config (database LLM)
        print("📋 DEFAULT CONFIG (Including Database LLM):")
        config_dict = default_config.to_dict()
        for key, value in config_dict.items():
            if value is not None:
                # Special formatting for commercial_mode
                if key == "commercial_mode":
                    mode_text = "Commercial (licensed datasets only)" if value else "Academic (all datasets)"
                    print(f"  {key.replace('_', ' ').title()}: {mode_text}")
                else:
                    print(f"  {key.replace('_', ' ').title()}: {value}")

        # Show agent-specific LLM if different from default
        if agent_llm != default_config.llm or agent_source != default_config.source:
            print("\n🤖 AGENT LLM (Constructor Override):")
            print(f"  LLM Model: {agent_llm}")
            if agent_source is not None:
                print(f"  Source: {agent_source}")
            if base_url is not None:
                print(f"  Base URL: {base_url}")
            if api_key is not None and api_key != "EMPTY":
                print(f"  API Key: {'*' * 8 + api_key[-4:] if len(api_key) > 8 else '***'}")

        print("=" * 50 + "\n")

        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Created directory: {path}")

        # --- Begin custom folder/file checks ---
        benchmark_dir = os.path.join(path, "biomni_data", "benchmark")
        data_lake_dir = os.path.join(path, "biomni_data", "data_lake")

        # Create the biomni_data directory structure
        os.makedirs(benchmark_dir, exist_ok=True)
        os.makedirs(data_lake_dir, exist_ok=True)

        if expected_data_lake_files is None:
            expected_data_lake_files = list(self.data_lake_dict.keys())

            # Check and download missing data lake files
            print("Checking and downloading missing data lake files...")
            check_and_download_s3_files(
                s3_bucket_url="https://biomni-release.s3.amazonaws.com",
                local_data_lake_path=data_lake_dir,
                expected_files=expected_data_lake_files,
                folder="data_lake",
            )

            # Check if benchmark directory structure is complete
            benchmark_ok = False
            if os.path.isdir(benchmark_dir):
                patient_gene_detection_dir = os.path.join(benchmark_dir, "hle")
                if os.path.isdir(patient_gene_detection_dir):
                    benchmark_ok = True

            if not benchmark_ok:
                print("Checking and downloading benchmark files...")
                check_and_download_s3_files(
                    s3_bucket_url="https://biomni-release.s3.amazonaws.com",
                    local_data_lake_path=benchmark_dir,
                    expected_files=[],  # Empty list - will download entire folder
                    folder="benchmark",
                )
        else:
            print("Skipping datalake download (load_datalake=False)")
            print("Note: Some tools may require datalake files to function properly.")

        self.path = os.path.join(path, "biomni_data")
        module2api = read_module2api()

        self.llm = get_llm(
            llm,
            stop_sequences=["</execute>", "</solution>"],
            source=source,
            base_url=base_url,
            api_key=api_key,
            config=default_config,
        )
        self.llm_model_name = llm
        self.llm_source = source
        self.llm_base_url = base_url
        self.llm_api_key = None if api_key in (None, "EMPTY") else api_key
        self.module2api = module2api
        self.use_tool_retriever = use_tool_retriever

        if self.use_tool_retriever:
            self.tool_registry = ToolRegistry(module2api)
            self.retriever = ToolRetriever()

        # Initialize know-how loader
        self.know_how_loader = KnowHowLoader()

        # Filter know-how documents based on commercial mode
        if commercial_mode:
            self._filter_know_how_for_commercial_mode()

        print(f"📚 Loaded {len(self.know_how_loader.documents)} know-how documents")

        # Add timeout parameter
        self.timeout_seconds = timeout_seconds  # 10 minutes default timeout
        self.configure()

    def update_runtime_llm_config(
        self,
        provider_name: str,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> dict[str, str]:
        resolved = _gradio_resolve_runtime_llm_settings(
            provider_name,
            model_name=model,
            base_url=base_url or "",
        )
        resolved_provider = resolved["provider"]
        resolved_model = resolved["model"]
        if not resolved_model:
            raise ValueError("Model name is required.")

        resolved_base_url = resolved["base_url"]
        if not resolved_base_url:
            raise ValueError("Base URL is required for runtime model switching.")

        resolved_api_key = (api_key or "").strip() or self.llm_api_key or default_config.api_key
        if not resolved_api_key:
            raise ValueError("API key is required before applying a new model.")

        self.llm = get_llm(
            resolved_model,
            stop_sequences=["</execute>", "</solution>"],
            source="Custom",
            base_url=resolved_base_url,
            api_key=resolved_api_key,
            config=default_config,
        )
        self.llm_model_name = resolved_model
        self.llm_source = "Custom"
        self.llm_base_url = resolved_base_url
        self.llm_api_key = resolved_api_key

        default_config.llm = resolved_model
        default_config.source = "Custom"
        default_config.base_url = resolved_base_url
        default_config.api_key = resolved_api_key

        os.environ["BIOMNI_LLM"] = resolved_model
        os.environ["BIOMNI_SOURCE"] = "Custom"
        os.environ["BIOMNI_CUSTOM_BASE_URL"] = resolved_base_url
        os.environ["BIOMNI_CUSTOM_API_KEY"] = resolved_api_key

        self.configure()
        return {
            "provider": resolved_provider,
            "source": "Custom",
            "model": resolved_model,
            "base_url": resolved_base_url,
        }

    def add_tool(self, api):
        """Add a new tool to the agent's tool registry and make it available for retrieval.

        Args:
            api: A callable function to be added as a tool

        """
        try:
            # Get function information
            function_code = inspect.getsource(api)
            module_name = api.__module__ if hasattr(api, "__module__") else "custom_tools"
            function_name = api.__name__ if hasattr(api, "__name__") else str(api)

            # Generate API schema using the existing utility function
            schema = function_to_api_schema(function_code, self.llm)

            # Ensure the schema has all required fields for the tool registry
            if not isinstance(schema, dict):
                raise ValueError("Generated schema is not a dictionary")

            # Validate and enhance the schema

            # Set default values if missing
            if "name" not in schema:
                schema["name"] = function_name
            if "description" not in schema:
                schema["description"] = f"Custom tool: {function_name}"
            if "required_parameters" not in schema:
                # Try to extract from parameters if available
                if "parameters" in schema and isinstance(schema["parameters"], dict):
                    required_params = []
                    params = schema["parameters"]
                    if "properties" in params:
                        for param_name in params["properties"]:
                            if param_name in params.get("required", []):
                                required_params.append(param_name)
                    schema["required_parameters"] = required_params
                else:
                    schema["required_parameters"] = []

            # Add module information to the schema
            schema["module"] = module_name

            # Add the tool to the tool registry if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    self.tool_registry.register_tool(schema)
                    print(f"Successfully registered tool '{schema['name']}' in tool registry")
                except Exception as e:
                    print(f"Warning: Failed to register tool in registry: {e}")
                    # Continue with adding to module2api even if registry fails

            # Add the tool to module2api structure for system prompt generation
            if not hasattr(self, "module2api") or self.module2api is None:
                self.module2api = {}

            if module_name not in self.module2api:
                self.module2api[module_name] = []

            # Check if tool already exists in module2api to avoid duplicates
            existing_tool = None
            for existing in self.module2api[module_name]:
                if existing.get("name") == schema["name"]:
                    existing_tool = existing
                    break

            if existing_tool:
                # Update existing tool
                existing_tool.update(schema)
                print(f"Updated existing tool '{schema['name']}' in module '{module_name}'")
            else:
                # Add new tool
                self.module2api[module_name].append(schema)
                print(f"Added new tool '{schema['name']}' to module '{module_name}'")

            # Update the tool registry's document dataframe if it exists
            if hasattr(self, "tool_registry") and self.tool_registry is not None:
                try:
                    # Rebuild the document dataframe
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

            # Store the original function for potential future use
            if not hasattr(self, "_custom_functions"):
                self._custom_functions = {}
            self._custom_functions[schema["name"]] = api

            # Also store in _custom_tools for highlighting
            if not hasattr(self, "_custom_tools"):
                self._custom_tools = {}
            self._custom_tools[schema["name"]] = {
                "name": schema["name"],
                "description": schema["description"],
                "module": module_name,
            }

            # Make the function available in the global namespace for execution
            import builtins

            if not hasattr(builtins, "_biomni_custom_functions"):
                builtins._biomni_custom_functions = {}
            builtins._biomni_custom_functions[schema["name"]] = api

            print(
                f"Tool '{schema['name']}' successfully added and ready for use in both direct execution and retrieval"
            )
            self.configure()
            return schema

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback

            traceback.print_exc()
            raise

    def add_mcp(self, config_path: str | Path = "./tutorials/examples/mcp_config.yaml") -> None:
        """
        Add MCP (Model Context Protocol) tools from configuration file.

        This method dynamically registers MCP server tools as callable functions within
        the biomni agent system. Each MCP server is loaded as an independent module
        with its tools exposed as synchronous wrapper functions.

        Supports both manual tool definitions and automatic tool discovery from MCP servers.

        Args:
            config_path: Path to the MCP configuration YAML file containing server
                        definitions and tool specifications.

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is malformed
            RuntimeError: If MCP server initialization fails
        """
        import asyncio
        import os
        import sys
        import types
        from pathlib import Path

        import nest_asyncio
        import yaml
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        nest_asyncio.apply()

        def discover_mcp_tools_sync(server_params: StdioServerParameters) -> list[dict]:
            """Discover available tools from MCP server synchronously."""
            try:

                async def _discover_async():
                    async with stdio_client(server_params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()

                            # Get available tools
                            tools_result = await session.list_tools()
                            tools = tools_result.tools if hasattr(tools_result, "tools") else tools_result

                            discovered_tools = []
                            for tool in tools:
                                if hasattr(tool, "name"):
                                    discovered_tools.append(
                                        {
                                            "name": tool.name,
                                            "description": tool.description,
                                            "inputSchema": tool.inputSchema,
                                        }
                                    )
                                else:
                                    print(f"Warning: Skipping tool with no name attribute: {tool}")

                            return discovered_tools

                return asyncio.run(_discover_async())
            except Exception as e:
                print(f"Failed to discover tools: {e}")
                return []

        def make_mcp_wrapper(cmd: str, args: list[str], tool_name: str, doc: str, env_vars: dict = None):
            """Create a synchronous wrapper for an async MCP tool call."""

            def sync_tool_wrapper(**kwargs):
                """Synchronous wrapper for MCP tool execution."""
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)

                    async def async_tool_call():
                        async with stdio_client(server_params) as (reader, writer):
                            async with ClientSession(reader, writer) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, kwargs)
                                content = result.content[0]
                                if hasattr(content, "json"):
                                    return content.json()
                                return content.text

                    try:
                        loop = asyncio.get_running_loop()
                        return loop.create_task(async_tool_call())
                    except RuntimeError:
                        return asyncio.run(async_tool_call())

                except Exception as e:
                    raise RuntimeError(f"MCP tool execution failed for '{tool_name}': {e}") from e

            sync_tool_wrapper.__name__ = tool_name
            sync_tool_wrapper.__doc__ = doc
            return sync_tool_wrapper

        # Initialize registries if they don't exist
        self._custom_functions = getattr(self, "_custom_functions", {})
        self._custom_tools = getattr(self, "_custom_tools", {})

        # Load and validate configuration
        try:
            config_content = Path(config_path).read_text(encoding="utf-8")
            cfg: dict[str, Any] = yaml.safe_load(config_content) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP config file not found: {config_path}") from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in MCP config: {e}") from e

        mcp_servers: dict[str, Any] = cfg.get("mcp_servers", {})
        if not mcp_servers:
            print("Warning: No MCP servers found in configuration")
            return

        # Process each MCP server configuration
        for server_name, server_meta in mcp_servers.items():
            if not server_meta.get("enabled", True):
                continue

            # Validate command configuration
            cmd_list = server_meta.get("command", [])
            if not cmd_list or not isinstance(cmd_list, list):
                print(f"Warning: Invalid command configuration for server '{server_name}'")
                continue

            cmd, *args = cmd_list

            # Process environment variables
            env_vars = server_meta.get("env", {})
            if env_vars:
                processed_env = {}
                for key, value in env_vars.items():
                    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                        var_name = value[2:-1]
                        processed_env[key] = os.getenv(var_name, "")
                    else:
                        processed_env[key] = value
                env_vars = processed_env

            # Create module namespace for this MCP server
            mcp_module_name = f"mcp_servers.{server_name}"
            if mcp_module_name not in sys.modules:
                sys.modules[mcp_module_name] = types.ModuleType(mcp_module_name)
            server_module = sys.modules[mcp_module_name]

            tools_config = server_meta.get("tools", [])

            if not tools_config:
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)
                    tools_config = discover_mcp_tools_sync(server_params)

                    if tools_config:
                        print(f"Discovered {len(tools_config)} tools from {server_name} MCP server")
                    else:
                        print(f"Warning: No tools discovered from {server_name} MCP server")
                        continue

                except Exception as e:
                    print(f"Failed to discover tools for {server_name}: {e}")
                    continue

            # Register each tool
            for tool_meta in tools_config:
                if isinstance(tool_meta, dict) and "biomni_name" in tool_meta:
                    # Manual tool definition
                    tool_name = tool_meta.get("biomni_name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    parameters = tool_meta.get("parameters", {})
                    # For manual tools, check if each parameter has a "required" field
                    required_param_names = []
                    for param_name, param_spec in parameters.items():
                        if param_spec.get("required", False):
                            required_param_names.append(param_name)
                else:
                    # Auto-discovered tool
                    tool_name = tool_meta.get("name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    input_schema = tool_meta.get("inputSchema", {})
                    parameters = input_schema.get("properties", {})
                    # For auto-discovered tools, get required list from inputSchema top level
                    required_param_names = input_schema.get("required", [])

                if not tool_name:
                    print(f"Warning: Skipping tool with no name in {server_name}")
                    continue

                # Create wrapper function
                wrapper_function = make_mcp_wrapper(cmd, args, tool_name, description, env_vars)

                # Add to module namespace
                setattr(server_module, tool_name, wrapper_function)

                # Build parameter lists
                required_params, optional_params = [], []
                for param_name, param_spec in parameters.items():
                    param_info = {
                        "name": param_name,
                        "type": str(param_spec.get("type", "string")),
                        "description": param_spec.get("description", ""),
                        "default": param_spec.get("default", None),
                    }

                    # Check if parameter is required based on the required_param_names list
                    if param_name in required_param_names:
                        required_params.append(param_info)
                    else:
                        optional_params.append(param_info)

                # Create tool schema
                tool_schema = {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                    "required_parameters": required_params,
                    "optional_parameters": optional_params,
                    "module": mcp_module_name,
                    "fn": wrapper_function,
                }

                # Register in tool registry
                self.tool_registry.register_tool(tool_schema)

                # Add to module2api mapping
                if mcp_module_name not in self.module2api:
                    self.module2api[mcp_module_name] = []
                self.module2api[mcp_module_name].append(tool_schema)

                # Add to instance registries
                self._custom_functions[tool_name] = wrapper_function
                self._custom_tools[tool_name] = {
                    "name": tool_name,
                    "description": description,
                    "module": mcp_module_name,
                }

        # Update agent configuration
        self.configure()

    def get_custom_tool(self, name):
        """Get a custom tool by name.

        Args:
            name: The name of the custom tool

        Returns:
            The custom tool function if found, None otherwise

        """
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            return self._custom_functions[name]
        return None

    def list_custom_tools(self):
        """List all custom tools that have been added.

        Returns:
            A list of custom tool names

        """
        if hasattr(self, "_custom_functions"):
            return list(self._custom_functions.keys())
        return []

    def remove_custom_tool(self, name):
        """Remove a custom tool.

        Args:
            name: The name of the custom tool to remove

        Returns:
            True if the tool was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom functions
        if hasattr(self, "_custom_functions") and name in self._custom_functions:
            del self._custom_functions[name]
            removed = True

        # Remove from custom tools (for highlighting)
        if hasattr(self, "_custom_tools") and name in self._custom_tools:
            del self._custom_tools[name]
            removed = True

        # Remove from global namespace
        import builtins

        if hasattr(builtins, "_biomni_custom_functions") and name in builtins._biomni_custom_functions:
            del builtins._biomni_custom_functions[name]

        # Remove from tool registry
        if hasattr(self, "tool_registry") and self.tool_registry is not None:
            if self.tool_registry.remove_tool_by_name(name):
                removed = True
                # Rebuild the document dataframe
                try:
                    docs = []
                    for tool_id in range(len(self.tool_registry.tools)):
                        docs.append(
                            [
                                int(tool_id),
                                self.tool_registry.get_tool_by_id(int(tool_id)),
                            ]
                        )
                    self.tool_registry.document_df = pd.DataFrame(docs, columns=["docid", "document_content"])
                except Exception as e:
                    print(f"Warning: Failed to update tool registry document dataframe: {e}")

        # Remove from module2api
        if hasattr(self, "module2api"):
            for tools in self.module2api.values():
                for i, tool in enumerate(tools):
                    if tool.get("name") == name:
                        del tools[i]
                        removed = True
                        break

        if removed:
            print(f"Custom tool '{name}' has been removed")
        else:
            print(f"Custom tool '{name}' was not found")

        return removed

    def add_data(self, data):
        """Add new data to the data lake.

        Args:
            data: Dictionary with file path as key and description as value
                  e.g., {'my_dataset.csv': 'A dataset containing gene expression data'}
                  or {'path/to/file.txt': 'Description of the file'}

        """
        try:
            if not isinstance(data, dict):
                raise ValueError("Data must be a dictionary with file path as key and description as value")

            # Initialize custom data storage if it doesn't exist
            if not hasattr(self, "_custom_data"):
                self._custom_data = {}

            # Add each data item
            for file_path, description in data.items():
                if not isinstance(file_path, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid data entry - file_path and description must be strings")
                    continue

                # Extract filename from path for storage
                filename = os.path.basename(file_path) if "/" in file_path else file_path

                # Store the data with both the full path and description
                self._custom_data[filename] = {
                    "path": file_path,
                    "description": description,
                }

                # Also add to the data_lake_dict for consistency
                self.data_lake_dict[filename] = description

                print(f"Added data item '{filename}': {description}")
            self.configure()
            print(f"Successfully added {len(data)} data item(s) to the data lake")
            return True

        except Exception as e:
            print(f"Error adding data: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_data(self, name):
        """Get a custom data item by name.

        Args:
            name: The name of the custom data item

        Returns:
            The custom data item info if found, None otherwise

        """
        if hasattr(self, "_custom_data") and name in self._custom_data:
            return self._custom_data[name]
        return None

    def list_custom_data(self):
        """List all custom data items that have been added.

        Returns:
            A list of custom data item names and descriptions

        """
        if hasattr(self, "_custom_data"):
            return [(name, info["description"]) for name, info in self._custom_data.items()]
        return []

    def remove_custom_data(self, name):
        """Remove a custom data item.

        Args:
            name: The name of the custom data item to remove

        Returns:
            True if the data item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom data
        if hasattr(self, "_custom_data") and name in self._custom_data:
            del self._custom_data[name]
            removed = True

        # Remove from data_lake_dict
        if hasattr(self, "data_lake_dict") and name in self.data_lake_dict:
            del self.data_lake_dict[name]
            removed = True

        if removed:
            print(f"Custom data item '{name}' has been removed")
        else:
            print(f"Custom data item '{name}' was not found")

        return removed

    def add_software(self, software):
        """Add new software to the software library.

        Args:
            software: Dictionary with software name as key and description as value
                     e.g., {'custom_tool': 'A custom analysis tool for processing data'}
                     or {'my_package': 'Description of the package functionality'}

        """
        try:
            if not isinstance(software, dict):
                raise ValueError("Software must be a dictionary with software name as key and description as value")

            # Initialize custom software storage if it doesn't exist
            if not hasattr(self, "_custom_software"):
                self._custom_software = {}

            # Add each software item
            for software_name, description in software.items():
                if not isinstance(software_name, str) or not isinstance(description, str):
                    print("Warning: Skipping invalid software entry - software_name and description must be strings")
                    continue

                # Store the software with description
                self._custom_software[software_name] = {
                    "name": software_name,
                    "description": description,
                }

                # Also add to the library_content_dict for consistency
                self.library_content_dict[software_name] = description

                print(f"Added software '{software_name}': {description}")

            print(f"Successfully added {len(software)} software item(s) to the library")
            self.configure()
            return True

        except Exception as e:
            print(f"Error adding software: {e}")
            import traceback

            traceback.print_exc()
            return False

    def get_custom_software(self, name):
        """Get a custom software item by name.

        Args:
            name: The name of the custom software item

        Returns:
            The custom software item info if found, None otherwise

        """
        if hasattr(self, "_custom_software") and name in self._custom_software:
            return self._custom_software[name]
        return None

    def list_custom_software(self):
        """List all custom software items that have been added.

        Returns:
            A list of custom software item names and descriptions

        """
        if hasattr(self, "_custom_software"):
            return [(name, info["description"]) for name, info in self._custom_software.items()]
        return []

    def remove_custom_software(self, name):
        """Remove a custom software item.

        Args:
            name: The name of the custom software item to remove

        Returns:
            True if the software item was removed, False if it wasn't found

        """
        removed = False

        # Remove from custom software
        if hasattr(self, "_custom_software") and name in self._custom_software:
            del self._custom_software[name]
            removed = True

        # Remove from library_content_dict
        if hasattr(self, "library_content_dict") and name in self.library_content_dict:
            del self.library_content_dict[name]
            removed = True

        if removed:
            print(f"Custom software item '{name}' has been removed")
        else:
            print(f"Custom software item '{name}' was not found")

        return removed

    def _filter_know_how_for_commercial_mode(self):
        """Filter out know-how documents that don't allow commercial use.

        This method removes documents from the know-how loader that have
        commercial use restrictions when the agent is in commercial mode.
        """
        docs_to_remove = []

        for doc_id, doc in self.know_how_loader.documents.items():
            metadata = doc.get("metadata", {})
            commercial_use = metadata.get("commercial_use", "")

            # Check if commercial use is NOT allowed
            if "❌" in commercial_use or "Not Allowed" in commercial_use or "Non-Commercial" in commercial_use:
                docs_to_remove.append(doc_id)

        # Remove documents that don't allow commercial use
        for doc_id in docs_to_remove:
            doc_name = self.know_how_loader.documents[doc_id]["name"]
            self.know_how_loader.remove_document(doc_id)
            print(f"  ⚠️  Excluded know-how '{doc_name}' (non-commercial license)")

    def _generate_system_prompt(
        self,
        tool_desc,
        data_lake_content,
        library_content_list,
        self_critic=False,
        is_retrieval=False,
        custom_tools=None,
        custom_data=None,
        custom_software=None,
        know_how_docs=None,
    ):
        """Generate the system prompt based on the provided resources.

        Args:
            tool_desc: Dictionary of tool descriptions
            data_lake_content: List of data lake items
            library_content_list: List of libraries
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)
            custom_tools: List of custom tools to highlight
            custom_data: List of custom data items to highlight
            custom_software: List of custom software items to highlight
            know_how_docs: List of know-how documents with best practices and protocols

        Returns:
            The generated system prompt

        """

        def format_item_with_description(name, description):
            """Format an item with its description in a readable way."""
            # Handle None or empty descriptions
            if not description:
                description = f"Data lake item: {name}"

            # Check if the item is already formatted (contains a colon)
            if isinstance(name, str) and ": " in name:
                return name

            # Wrap long descriptions to make them more readable
            max_line_length = 80
            if len(description) > max_line_length:
                # Simple wrapping for long descriptions
                wrapped_desc = []
                words = description.split()
                current_line = ""

                for word in words:
                    if len(current_line) + len(word) + 1 <= max_line_length:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                    else:
                        wrapped_desc.append(current_line)
                        current_line = word

                if current_line:
                    wrapped_desc.append(current_line)

                # Join with newlines and proper indentation
                formatted_desc = f"{name}:\n  " + "\n  ".join(wrapped_desc)
                return formatted_desc
            else:
                return f"{name}: {description}"

        # Separate custom and default resources
        default_data_lake_content = []
        default_library_content_list = []

        # Filter out custom items from default lists
        custom_data_names = set()
        custom_software_names = set()

        if custom_data:
            custom_data_names = {item.get("name") if isinstance(item, dict) else item for item in custom_data}
        if custom_software:
            custom_software_names = {item.get("name") if isinstance(item, dict) else item for item in custom_software}

        # Separate default data lake items
        for item in data_lake_content:
            if isinstance(item, dict):
                name = item.get("name", "")
                if name not in custom_data_names:
                    default_data_lake_content.append(item)
            elif item not in custom_data_names:
                default_data_lake_content.append(item)

        # Separate default library items
        for lib in library_content_list:
            if isinstance(lib, dict):
                name = lib.get("name", "")
                if name not in custom_software_names:
                    default_library_content_list.append(lib)
            elif lib not in custom_software_names:
                default_library_content_list.append(lib)

        # Format the default data lake content
        if isinstance(default_data_lake_content, list) and all(
            isinstance(item, str) for item in default_data_lake_content
        ):
            # Simple list of strings - check if they already have descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                # Check if the item already has a description (contains a colon)
                if ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))
        else:
            # List with descriptions
            data_lake_formatted = []
            for item in default_data_lake_content:
                if isinstance(item, dict):
                    name = item.get("name", "")
                    description = self.data_lake_dict.get(name, f"Data lake item: {name}")
                    data_lake_formatted.append(format_item_with_description(name, description))
                # Check if the item already has a description (contains a colon)
                elif isinstance(item, str) and ": " in item:
                    data_lake_formatted.append(item)
                else:
                    description = self.data_lake_dict.get(item, f"Data lake item: {item}")
                    data_lake_formatted.append(format_item_with_description(item, description))

        # Format the default library content
        if isinstance(default_library_content_list, list) and all(
            isinstance(item, str) for item in default_library_content_list
        ):
            if (
                len(default_library_content_list) > 0
                and isinstance(default_library_content_list[0], str)
                and "," not in default_library_content_list[0]
            ):
                # Simple list of strings
                libraries_formatted = []
                for lib in default_library_content_list:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))
            else:
                # Already formatted string
                libraries_formatted = default_library_content_list
        else:
            # List with descriptions
            libraries_formatted = []
            for lib in default_library_content_list:
                if isinstance(lib, dict):
                    name = lib.get("name", "")
                    description = self.library_content_dict.get(name, f"Software library: {name}")
                    libraries_formatted.append(format_item_with_description(name, description))
                else:
                    description = self.library_content_dict.get(lib, f"Software library: {lib}")
                    libraries_formatted.append(format_item_with_description(lib, description))

        # Format custom resources with highlighting
        custom_tools_formatted = []
        if custom_tools:
            for tool in custom_tools:
                if isinstance(tool, dict):
                    name = tool.get("name", "Unknown")
                    desc = tool.get("description", "")
                    module = tool.get("module", "custom_tools")
                    custom_tools_formatted.append(f"🔧 {name} (from {module}): {desc}")
                else:
                    custom_tools_formatted.append(f"🔧 {str(tool)}")

        custom_data_formatted = []
        if custom_data:
            for item in custom_data:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_data_formatted.append(f"📊 {format_item_with_description(name, desc)}")
                else:
                    desc = self.data_lake_dict.get(item, f"Custom data: {item}")
                    custom_data_formatted.append(f"📊 {format_item_with_description(item, desc)}")

        custom_software_formatted = []
        if custom_software:
            for item in custom_software:
                if isinstance(item, dict):
                    name = item.get("name", "Unknown")
                    desc = item.get("description", "")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(name, desc)}")
                else:
                    desc = self.library_content_dict.get(item, f"Custom software: {item}")
                    custom_software_formatted.append(f"⚙️ {format_item_with_description(item, desc)}")

        # Format know-how documents - include FULL content (metadata already stripped)
        know_how_formatted = []
        if know_how_docs:
            for doc in know_how_docs:
                if isinstance(doc, dict):
                    name = doc.get("name", "Unknown")
                    content = doc.get("content", "")
                    # Include full content in system prompt (metadata already removed)
                    know_how_formatted.append(f"📚 {name}:\n{content}")

        # Base prompt
        prompt_modifier = """
You are a helpful biomedical assistant assigned with the task of problem-solving.
To achieve this, you will be using an interactive coding environment equipped with a variety of tool functions, data, and softwares to assist you throughout the process.

Given a task, make a plan first. The plan should be a numbered list of steps that you will take to solve the task. Be specific and detailed.
Format your plan as a checklist with empty checkboxes like this:
1. [ ] First step
2. [ ] Second step
3. [ ] Third step

Follow the plan step by step. After completing each step, update the checklist by replacing the empty checkbox with a checkmark:
1. [✓] First step (completed)
2. [ ] Second step
3. [ ] Third step

If a step fails or needs modification, mark it with an X and explain why:
1. [✓] First step (completed)
2. [✗] Second step (failed because...)
3. [ ] Modified second step
4. [ ] Third step

Always show the updated plan after each step so the user can track progress.

At each turn, you should first provide your thinking and reasoning given the conversation history.
After that, you have two options:

1) Interact with a programming environment and receive the corresponding output within <observe></observe>. Your code should be enclosed using "<execute>" tag, for example: <execute> print("Hello World!") </execute>. IMPORTANT: You must end the code block with </execute> tag.
   - For Python code (default): <execute> print("Hello World!") </execute>
   - For R code: <execute> #!R\nlibrary(ggplot2)\nprint("Hello from R") </execute>
   - For Bash scripts and commands: <execute> #!BASH\necho "Hello from Bash"\nls -la </execute>
   - For CLI softwares, use Bash scripts.

2) When you think it is ready, directly provide a solution that adheres to the required format for the given task to the user. Your solution should be enclosed using "<solution>" tag, for example: The answer is <solution> A </solution>. IMPORTANT: You must end the solution block with </solution> tag.

You have many chances to interact with the environment to receive the observation. So you can decompose your code into multiple steps.
Don't overcomplicate the code. Keep it simple and easy to understand.
When writing the code, please print out the steps and results in a clear and concise manner, like a research log.
When calling the existing python functions in the function dictionary, YOU MUST SAVE THE OUTPUT and PRINT OUT the result.
For example, result = understand_scRNA(XXX) print(result)
Otherwise the system will not be able to know what has been done.

For R code, use the #!R marker at the beginning of your code block to indicate it's R code.
For Bash scripts and commands, use the #!BASH marker at the beginning of your code block. This allows for both simple commands and multi-line scripts with variables, loops, conditionals, loops, and other Bash features.

In each response, you must include EITHER <execute> or <solution> tag. Not both at the same time. Do not respond with messages without any tags. No empty messages.
"""

        # Add self-critic instructions if needed
        if self_critic:
            prompt_modifier += """
You may or may not receive feedbacks from human. If so, address the feedbacks by following the same procedure of multiple rounds of thinking, execution, and then coming up with a new solution.
"""

        # Add protocol generation instructions
        prompt_modifier += """
PROTOCOL GENERATION:
If the user requests an experimental protocol, use search_protocols(), advanced_web_search_claude(), list_local_protocols(), and read_local_protocol() to generate an accurate protocol. Include details such as reagents (with catalog numbers if available), equipment specifications, replicate requirements, error handling, and troubleshooting - but ONLY include information found in these resources. Do not make up specifications, catalog numbers, or equipment details. Prioritize accuracy over completeness.
"""

        # Add custom resources section first (highlighted)
        has_custom_resources = any(
            [custom_tools_formatted, custom_data_formatted, custom_software_formatted, know_how_formatted]
        )

        if has_custom_resources:
            prompt_modifier += """

PRIORITY CUSTOM RESOURCES
===============================
IMPORTANT: The following custom resources have been specifically added for your use.
    PRIORITIZE using these resources as they are directly relevant to your task.
    Always consider these FIRST and in the meantime using default resources.

"""

            if know_how_formatted:
                prompt_modifier += """
📚 KNOW-HOW DOCUMENTS (BEST PRACTICES & PROTOCOLS - ALREADY LOADED):
{know_how_docs}

IMPORTANT: These documents are ALREADY AVAILABLE in your context. You do NOT need to
retrieve them or "review" them as a separate step. You can DIRECTLY reference and use
the information from these documents to answer questions, provide protocols, suggest
parameters, and offer troubleshooting guidance.

These documents contain expert knowledge, protocols, and troubleshooting guidance.
Reference them directly for experimental design, methodology, and problem-solving.

"""

            if custom_tools_formatted:
                prompt_modifier += """
🔧 CUSTOM TOOLS (USE THESE FIRST):
{custom_tools}

"""

            if custom_data_formatted:
                prompt_modifier += """
📊 CUSTOM DATA (PRIORITIZE THESE DATASETS):
{custom_data}

"""

            if custom_software_formatted:
                prompt_modifier += """
⚙️ CUSTOM SOFTWARE (USE THESE LIBRARIES):
{custom_software}

"""

            prompt_modifier += """===============================
"""

        # Add environment resources
        prompt_modifier += """

Environment Resources:

- Function Dictionary:
{function_intro}
---
{tool_desc}
---

{import_instruction}

- Biological data lake
You can access a biological data lake at the following path: {data_lake_path}.
{data_lake_intro}
Each item is listed with its description to help you understand its contents.
----
{data_lake_content}
----

- Software Library:
{library_intro}
Each library is listed with its description to help you understand its functionality.
----
{library_content_formatted}
----

- Note on using R packages and Bash scripts:
  - R packages: Use subprocess.run(['Rscript', '-e', 'your R code here']) in Python, or use the #!R marker in your execute block.
  - Bash scripts and commands: Use the #!BASH marker in your execute block for both simple commands and complex shell scripts with variables, loops, conditionals, etc.
        """

        # Set appropriate text based on whether this is initial configuration or after retrieval
        if is_retrieval:
            function_intro = "Based on your query, I've identified the following most relevant functions that you can use in your code:"
            data_lake_intro = "Based on your query, I've identified the following most relevant datasets:"
            library_intro = (
                "Based on your query, I've identified the following most relevant libraries that you can use:"
            )
            import_instruction = "IMPORTANT: When using any function, you MUST first import it from its module. For example:\nfrom [module_name] import [function_name]"
        else:
            function_intro = "In your code, you will need to import the function location using the following dictionary of functions:"
            data_lake_intro = "You can write code to understand the data, process and utilize it for the task. Here is the list of datasets:"
            library_intro = "The environment supports a list of libraries that can be directly used. Do not forget the import statement:"
            import_instruction = ""

        # Format the content consistently for both initial and retrieval cases
        library_content_formatted = "\n".join(libraries_formatted)
        data_lake_content_formatted = "\n".join(data_lake_formatted)

        # Format the prompt with the appropriate values
        format_dict = {
            "function_intro": function_intro,
            "tool_desc": textify_api_dict(tool_desc) if isinstance(tool_desc, dict) else tool_desc,
            "import_instruction": import_instruction,
            "data_lake_path": self.path + "/data_lake",
            "data_lake_intro": data_lake_intro,
            "data_lake_content": data_lake_content_formatted,
            "library_intro": library_intro,
            "library_content_formatted": library_content_formatted,
        }

        # Add custom resources to format dict if they exist
        if know_how_formatted:
            format_dict["know_how_docs"] = "\n\n".join(know_how_formatted)
        if custom_tools_formatted:
            format_dict["custom_tools"] = "\n".join(custom_tools_formatted)
        if custom_data_formatted:
            format_dict["custom_data"] = "\n".join(custom_data_formatted)
        if custom_software_formatted:
            format_dict["custom_software"] = "\n".join(custom_software_formatted)

        formatted_prompt = prompt_modifier.format(**format_dict)

        return formatted_prompt

    def configure(self, self_critic=False, test_time_scale_round=0):
        """Configure the agent with the initial system prompt and workflow.

        Args:
            self_critic: Whether to enable self-critic mode
            test_time_scale_round: Number of rounds for test time scaling

        """
        # Store self_critic for later use
        self.self_critic = self_critic

        # Get data lake content
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # data_lake_dict and library_content_dict are already set in __init__

        # Prepare tool descriptions
        tool_desc = {i: [x for x in j if x["name"] != "run_python_repl"] for i, j in self.module2api.items()}

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Add custom data items if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_with_desc.append({"name": name, "description": info["description"]})

        # Prepare library content list including custom software
        library_content_list = list(self.library_content_dict.keys())
        if hasattr(self, "_custom_software") and self._custom_software:
            for name in self._custom_software:
                if name not in library_content_list:  # Avoid duplicates
                    library_content_list.append(name)

        # Generate the system prompt for initial configuration (is_retrieval=False)
        # Prepare custom resources for highlighting
        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "module": info["module"],
                    }
                )

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        # Load ALL know-how documents into initial system prompt
        # This makes best practices always available, not just when retrieved
        know_how_docs = []
        if hasattr(self, "know_how_loader") and self.know_how_loader.documents:
            for _doc_id, doc in self.know_how_loader.documents.items():
                # Use content without metadata for efficiency
                know_how_docs.append(
                    {
                        "id": doc["id"],
                        "name": doc["name"],
                        "description": doc["description"],
                        "content": doc["content_without_metadata"],
                        "metadata": doc["metadata"],
                    }
                )
            print(f"📚 Loading {len(know_how_docs)} know-how documents into system prompt")

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=library_content_list,
            self_critic=self_critic,
            is_retrieval=False,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
            know_how_docs=know_how_docs if know_how_docs else None,
        )

        # Define the nodes
        def generate(state: AgentState) -> AgentState:
            # Add OpenAI-specific formatting reminders if using OpenAI models
            system_prompt = self.system_prompt
            if hasattr(self.llm, "model_name") and (
                "gpt" in str(self.llm.model_name).lower() or "openai" in str(type(self.llm)).lower()
            ):
                system_prompt += "\n\nIMPORTANT FOR GPT MODELS: You MUST use XML tags <execute> or <solution> in EVERY response. Do not use markdown code blocks (```) - use <execute> tags instead."

            messages = [SystemMessage(content=system_prompt)] + state["messages"]
            response = self.llm.invoke(messages)

            # Normalize Responses API content blocks (list of dicts) into a plain string
            content = response.content
            if isinstance(content, list):
                # Concatenate textual parts; ignore tool_use or other non-text blocks
                text_parts: list[str] = []
                for block in content:
                    try:
                        if isinstance(block, dict):
                            btype = block.get("type")
                            if btype in ("text", "output_text", "redacted_text"):
                                part = block.get("text") or block.get("content") or ""
                                if isinstance(part, str):
                                    text_parts.append(part)
                    except Exception:
                        # Be conservative; skip malformed blocks
                        continue
                msg = "".join(text_parts)
            else:
                # Fallback to string conversion for legacy content
                msg = str(content)

            # Enhanced parsing for better OpenAI compatibility
            # Check for incomplete tags and fix them
            if "<execute>" in msg and "</execute>" not in msg:
                msg += "</execute>"
            if "<solution>" in msg and "</solution>" not in msg:
                msg += "</solution>"
            if "<think>" in msg and "</think>" not in msg:
                msg += "</think>"

            # More flexible pattern matching for different LLM styles
            think_match = re.search(r"<think>(.*?)</think>", msg, re.DOTALL | re.IGNORECASE)
            execute_match = re.search(r"<execute>(.*?)</execute>", msg, re.DOTALL | re.IGNORECASE)
            answer_match = re.search(r"<solution>(.*?)</solution>", msg, re.DOTALL | re.IGNORECASE)

            # Alternative patterns for OpenAI models that might use different formatting
            if not execute_match:
                # Try to find code blocks that might be intended as execute blocks
                code_block_match = re.search(r"```(?:python|bash|r)?\s*(.*?)```", msg, re.DOTALL)
                if code_block_match and not answer_match:
                    # If we found a code block and no solution, treat it as execute
                    execute_match = code_block_match

            # Add the message to the state before checking for errors
            state["messages"].append(AIMessage(content=msg.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            elif think_match:
                state["next_step"] = "generate"
            else:
                print("parsing error...")

                error_count = sum(
                    1 for m in state["messages"] if isinstance(m, AIMessage) and "There are no tags" in m.content
                )

                if error_count >= 2:
                    # If we've already tried to correct the model twice, just end the conversation
                    print("Detected repeated parsing errors, ending conversation")
                    state["next_step"] = "end"
                    # Add a final message explaining the termination
                    state["messages"].append(
                        AIMessage(
                            content="Execution terminated due to repeated parsing errors. Please check your input and try again."
                        )
                    )
                else:
                    # Try to correct it
                    state["messages"].append(
                        HumanMessage(
                            content="Each response must include thinking process followed by either <execute> or <solution> tag. But there are no tags in the current response. Please follow the instruction, fix and regenerate the response again."
                        )
                    )
                    state["next_step"] = "generate"
            return state

        def execute(state: AgentState) -> AgentState:
            last_message = state["messages"][-1].content
            # Only add the closing tag if it's not already there
            if "<execute>" in last_message and "</execute>" not in last_message:
                last_message += "</execute>"

            execute_match = re.search(r"<execute>(.*?)</execute>", last_message, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)

                # Set timeout duration (10 minutes = 600 seconds)
                timeout = self.timeout_seconds

                # Check if the code is R code
                if (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    # Remove the R marker and run as R code
                    r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, count=1).strip()
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
                    # Handle both Bash scripts and CLI commands with the same function
                    if code.strip().startswith("#!CLI"):
                        # For CLI commands, extract the command and run it as a simple bash script
                        cli_command = re.sub(r"^#!CLI", "", code, count=1).strip()
                        # Remove any newlines to ensure it's a single command
                        cli_command = cli_command.replace("\n", " ")
                        result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
                    else:
                        # For Bash scripts, remove the marker and run as a bash script
                        bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, count=1).strip()
                        result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                # Otherwise, run as Python code
                else:
                    # Clear any previous plots before execution
                    self._clear_execution_plots()

                    # Inject custom functions into the Python execution environment
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                    # Plots are now captured directly in the execution entry above

                if len(result) > 10000:
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )

                # Store the execution result with the triggering message
                if not hasattr(self, "_execution_results"):
                    self._execution_results = []

                # Get any plots that were generated during this execution
                execution_plots = []
                try:
                    from biomni.tool.support_tools import get_captured_plots

                    current_plots = get_captured_plots()
                    execution_plots = current_plots.copy()
                except Exception as e:
                    print(f"Warning: Could not capture plots from execution: {e}")
                    execution_plots = []

                # Store the execution result with metadata
                execution_entry = {
                    "triggering_message": last_message,  # The AI message that contained <execute>
                    "images": execution_plots,  # Base64 encoded images from this execution
                    "timestamp": datetime.now().isoformat(),
                }
                self._execution_results.append(execution_entry)

                observation = f"\n<observation>{result}</observation>"
                state["messages"].append(AIMessage(content=observation.strip()))

            return state

        def routing_function(
            state: AgentState,
        ) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def routing_function_self_critic(
            state: AgentState,
        ) -> Literal["generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def execute_self_critic(state: AgentState) -> AgentState:
            if self.critic_count < test_time_scale_round:
                # Generate feedback based on message history
                messages = state["messages"]
                feedback_prompt = f"""
                Here is a reminder of what is the user requested: {self.user_task}
                Examine the previous executions, reaosning, and solutions.
                Critic harshly on what could be improved?
                Be specific and constructive.
                Think hard what are missing to solve the task.
                No question asked, just feedbacks.
                """
                feedback = self.llm.invoke(messages + [HumanMessage(content=feedback_prompt)])

                # Add feedback as a new message
                state["messages"].append(
                    HumanMessage(
                        content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                    )
                )
                self.critic_count += 1
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"

            return state

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)

        if self_critic:
            workflow.add_node("self_critic", execute_self_critic)
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={
                    "execute": "execute",
                    "generate": "generate",
                    "end": "self_critic",
                },
            )
            workflow.add_conditional_edges(
                "self_critic",
                routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )
        else:
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")

        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        # display(Image(self.app.get_graph().draw_mermaid_png()))

    def _prepare_resources_for_retrieval(self, prompt):
        """Prepare resources for retrieval and return selected resource names.

        Args:
            prompt: The user's query

        Returns:
            dict: Dictionary containing selected resource names for tools, data_lake, and libraries
        """
        if not self.use_tool_retriever:
            return None

        # Gather all available resources
        # 1. Tools from the registry
        all_tools = self.tool_registry.tools if hasattr(self, "tool_registry") else []

        # 2. Data lake items with descriptions
        data_lake_path = self.path + "/data_lake"
        data_lake_content = glob.glob(data_lake_path + "/*")
        data_lake_items = [x.split("/")[-1] for x in data_lake_content]

        # Create data lake descriptions for retrieval
        data_lake_descriptions = []
        for item in data_lake_items:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_descriptions.append({"name": item, "description": description})

        # Add custom data items to retrieval if they exist
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                data_lake_descriptions.append({"name": name, "description": info["description"]})

        # 3. Libraries with descriptions - use library_content_dict directly
        library_descriptions = []
        for lib_name, lib_desc in self.library_content_dict.items():
            library_descriptions.append({"name": lib_name, "description": lib_desc})

        # Add custom software items to retrieval if they exist
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                # Check if it's not already in the library descriptions to avoid duplicates
                if not any(lib["name"] == name for lib in library_descriptions):
                    library_descriptions.append({"name": name, "description": info["description"]})

        # 4. Know-how documents
        know_how_summaries = self.know_how_loader.get_document_summaries()

        # Use retrieval to get relevant resources
        resources = {
            "tools": all_tools,
            "data_lake": data_lake_descriptions,
            "libraries": library_descriptions,
            "know_how": know_how_summaries,
        }

        # Use prompt-based retrieval with the agent's LLM
        selected_resources = self.retriever.prompt_based_retrieval(prompt, resources, llm=self.llm)
        print("\n" + "=" * 60)
        print("🔍 RESOURCE RETRIEVAL")
        print("=" * 60)
        print("Using prompt-based retrieval with the agent's LLM")

        # Extract the names from the selected resources for the system prompt
        selected_resources_names = {
            "tools": selected_resources["tools"],
            "data_lake": [],
            "libraries": [lib["name"] if isinstance(lib, dict) else lib for lib in selected_resources["libraries"]],
            "know_how": [],
        }

        # Process data lake items to extract just the names
        for item in selected_resources["data_lake"]:
            if isinstance(item, dict):
                selected_resources_names["data_lake"].append(item["name"])
            elif isinstance(item, str) and ": " in item:
                # If the item already has a description, extract just the name
                name = item.split(": ")[0]
                selected_resources_names["data_lake"].append(name)
            else:
                selected_resources_names["data_lake"].append(item)

        # Process know-how documents - get the full content for selected documents
        if "know_how" in selected_resources and selected_resources["know_how"]:
            print("\n📚 Know-How Documents Retrieved:")
            for item in selected_resources["know_how"]:
                if isinstance(item, dict):
                    doc_id = item["id"]
                    doc = self.know_how_loader.get_document_by_id(doc_id)
                    if doc:
                        print(f"  ✓ {doc['name']}")
                        # Create a copy with content_without_metadata for agent context
                        doc_for_agent = {
                            "id": doc["id"],
                            "name": doc["name"],
                            "description": doc["description"],
                            "content": doc["content_without_metadata"],  # Use stripped version for agent
                            "metadata": doc["metadata"],
                        }
                        selected_resources_names["know_how"].append(doc_for_agent)
        else:
            print("\n📚 Know-How: None retrieved for this query")

        # Print summary of what was retrieved
        print("\n" + "-" * 60)
        print("📊 RETRIEVAL SUMMARY:")
        print(f"  🔧 Tools: {len(selected_resources_names['tools'])} selected")
        print(f"  📊 Data Lake: {len(selected_resources_names['data_lake'])} selected")
        print(f"  ⚙️  Libraries: {len(selected_resources_names['libraries'])} selected")
        print(f"  📚 Know-How: {len(selected_resources_names['know_how'])} selected")
        print("=" * 60 + "\n")

        return selected_resources_names

    def go(self, prompt):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query

        """
        self.critic_count = 0
        self.user_task = prompt

        if self.use_tool_retriever:
            selected_resources_names = self._prepare_resources_for_retrieval(prompt)
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state for markdown generation
        final_state = None

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
            final_state = s  # Store the latest state

        # Store the conversation state for markdown generation
        self._conversation_state = final_state

        return self.log, message.content

    def go_stream(self, prompt) -> Generator[dict, None, None]:
        """Execute the agent with the given prompt and return a generator that yields each step.

        This function returns a generator that yields each step of the agent's execution,
        allowing for real-time monitoring of the agent's progress.

        Args:
            prompt: The user's query

        Yields:
            dict: Each step of the agent's execution containing the current message and state
        """
        self.critic_count = 0
        self.user_task = prompt

        if self.use_tool_retriever:
            selected_resources_names = self._prepare_resources_for_retrieval(prompt)
            self.update_system_prompt_with_selected_resources(selected_resources_names)

        inputs = {"messages": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state for markdown generation
        final_state = None

        for s in self.app.stream(inputs, stream_mode="values", config=config):
            message = s["messages"][-1]
            out = pretty_print(message)
            self.log.append(out)
            final_state = s  # Store the latest state

            # Yield the current step
            yield {"output": out}

        # Store the conversation state for markdown generation
        self._conversation_state = final_state

    def update_system_prompt_with_selected_resources(self, selected_resources):
        """Update the system prompt with the selected resources."""
        # Extract tool descriptions for the selected tools
        tool_desc = {}
        for tool in selected_resources["tools"]:
            # Get the module name from the tool
            if isinstance(tool, dict):
                module_name = tool.get("module", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool.get("name"):
                                module_name = mod
                                # Update the tool with the module information
                                tool["module"] = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "biomni.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool["module"] = module_name
            else:
                module_name = getattr(tool, "module_name", None)

                # If module is not specified, try to find it in the module2api
                if not module_name and hasattr(self, "module2api"):
                    tool_name = getattr(tool, "name", str(tool))
                    for mod, apis in self.module2api.items():
                        for api in apis:
                            if api.get("name") == tool_name:
                                module_name = mod
                                # Set the module_name attribute
                                tool.module_name = module_name
                                break
                        if module_name:
                            break

                # If still not found, use a default
                if not module_name:
                    module_name = "biomni.tool.scRNA_tools"  # Default to scRNA_tools as a fallback
                    tool.module_name = module_name

            if module_name not in tool_desc:
                tool_desc[module_name] = []

            # Add the tool to the appropriate module
            if isinstance(tool, dict):
                # Ensure the module is included in the tool description
                if "module" not in tool:
                    tool["module"] = module_name
                tool_desc[module_name].append(tool)
            else:
                # Convert tool object to dictionary
                tool_dict = {
                    "name": getattr(tool, "name", str(tool)),
                    "description": getattr(tool, "description", ""),
                    "parameters": getattr(tool, "parameters", {}),
                    "module": module_name,  # Explicitly include the module
                }
                tool_desc[module_name].append(tool_dict)

        # Prepare data lake items with descriptions
        data_lake_with_desc = []
        for item in selected_resources["data_lake"]:
            description = self.data_lake_dict.get(item, f"Data lake item: {item}")
            data_lake_with_desc.append({"name": item, "description": description})

        # Prepare custom resources for highlighting
        custom_tools = []
        if hasattr(self, "_custom_tools") and self._custom_tools:
            for name, info in self._custom_tools.items():
                custom_tools.append(
                    {
                        "name": name,
                        "description": info["description"],
                        "module": info["module"],
                    }
                )

        custom_data = []
        if hasattr(self, "_custom_data") and self._custom_data:
            for name, info in self._custom_data.items():
                custom_data.append({"name": name, "description": info["description"]})

        custom_software = []
        if hasattr(self, "_custom_software") and self._custom_software:
            for name, info in self._custom_software.items():
                custom_software.append({"name": name, "description": info["description"]})

        # Extract know-how documents if present
        know_how_docs = selected_resources.get("know_how", [])

        self.system_prompt = self._generate_system_prompt(
            tool_desc=tool_desc,
            data_lake_content=data_lake_with_desc,
            library_content_list=selected_resources["libraries"],
            self_critic=getattr(self, "self_critic", False),
            is_retrieval=True,
            custom_tools=custom_tools if custom_tools else None,
            custom_data=custom_data if custom_data else None,
            custom_software=custom_software if custom_software else None,
            know_how_docs=know_how_docs if know_how_docs else None,
        )

        # Print the raw system prompt for debugging
        # print("\n" + "="*20 + " RAW SYSTEM PROMPT FROM AGENT " + "="*20)
        # print(self.system_prompt)
        # print("="*70 + "\n")

    def result_formatting(self, output_class, task_intention):
        self.format_check_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    (
                        "You are evaluateGPT, tasked with extract and parse the task output based on the history of an agent. "
                        "Review the entire history of messages provided. "
                        "Here is the task output requirement: \n"
                        f"'{task_intention.replace('{', '{{').replace('}', '}}')}'.\n"
                    ),
                ),
                ("placeholder", "{messages}"),
            ]
        )

        checker_llm = self.format_check_prompt | self.llm.with_structured_output(output_class)
        result = checker_llm.invoke({"messages": [("user", str(self.log))]}).dict()
        return result

    def _parse_tool_calls_from_code(self, code: str) -> list[str]:
        """Parse code to detect imported tools by looking for import statements.

        Args:
            code: The Python code to parse

        Returns:
            List of detected tool names
        """
        module2api = getattr(self, "module2api", {})
        custom_functions = getattr(self, "_custom_functions", {})
        return parse_tool_calls_from_code(code, module2api, custom_functions)

    def _parse_tool_calls_with_modules(self, code: str) -> list[tuple[str, str]]:
        """Parse code to detect imported tools and their modules.

        Args:
            code: The Python code to parse

        Returns:
            List of tuples (tool_name, module_name)
        """
        module2api = getattr(self, "module2api", {})
        custom_functions = getattr(self, "_custom_functions", {})
        return parse_tool_calls_with_modules(code, module2api, custom_functions)

    def _inject_custom_functions_to_repl(self):
        """Inject custom functions into the Python REPL execution environment.
        This makes custom tools available during code execution.
        """
        custom_functions = getattr(self, "_custom_functions", {})
        inject_custom_functions_to_repl(custom_functions)

    def create_mcp_server(self, tool_modules=None):
        """
        Create an MCP server object that exposes internal Biomni tools.
        This gives you control over when and how to run the server.

        Args:
            tool_modules: List of module names to expose (default: all in self.module2api)

        Returns:
            FastMCP server object that you can run manually
        """
        import importlib

        from mcp.server.fastmcp import FastMCP

        mcp = FastMCP("BiomniTools")
        modules = tool_modules or list(self.module2api.keys())

        registered_tools = 0

        for module_name in modules:
            try:
                # Import the actual module
                module = importlib.import_module(module_name)
                # Get tools for this module
                module_tools = self.module2api.get(module_name, [])

                for tool_schema in module_tools:
                    tool_name = tool_schema.get("name")
                    if not tool_name:
                        continue

                    try:
                        # Get the actual function
                        fn = getattr(module, tool_name, None)
                        if fn is None:
                            fn = getattr(self, "_custom_functions", {}).get(tool_name)

                        if fn is None:
                            print(f"Warning: Could not find function '{tool_name}' in module '{module_name}'")
                            continue

                        # Extract parameters from your specific schema format
                        required_params = tool_schema.get("required_parameters", [])
                        optional_params = tool_schema.get("optional_parameters", [])

                        # Generate the wrapper function
                        wrapper_func = self._generate_mcp_wrapper_from_biomni_schema(
                            fn, tool_name, required_params, optional_params
                        )

                        # Register with MCP
                        mcp.tool()(wrapper_func)
                        registered_tools += 1

                    except Exception as e:
                        print(f"Warning: Failed to register tool '{tool_name}': {e}")
                        continue

            except ImportError as e:
                print(f"Warning: Could not import module '{module_name}': {e}")
                continue

        print(f"Created MCP server with {registered_tools} tools")
        return mcp

    def save_conversation_history(self, filepath: str, include_images: bool = True, save_pdf: bool = True) -> None:
        """Save the complete conversation history as PDF only.

        This function generates and saves the complete conversation history from the agent's
        log and conversation state. It creates a temporary markdown file with formatted content
        including steps, code execution, observations, and optionally images, then converts it
        to PDF format. The markdown file is automatically cleaned up after PDF conversion.

        Args:
            filepath: Path where to save the PDF file (without extension). If the path doesn't
                    end with .pdf, it will be automatically appended.
            include_images: Whether to include captured plots and images in the output.
                          Defaults to True.
            save_pdf: Whether to save as PDF. Defaults to True. If False, no file is saved.

        Note:
            The function includes a 60-second timeout for PDF generation to prevent
            hanging. A temporary markdown file is created and automatically deleted.
        """
        import os
        import tempfile

        if not save_pdf:
            print("PDF saving is disabled. No file will be saved.")
            return

        # Ensure directory exists
        directory = os.path.dirname(filepath)
        if directory:  # Only create directory if it's not empty
            os.makedirs(directory, exist_ok=True)

        # Create PDF file path - use the user's filename and add .pdf extension
        if filepath.endswith(".pdf"):
            pdf_path = filepath
        else:
            # Remove any existing .md extension if present, then add .pdf
            base_name = filepath
            if base_name.endswith(".md"):
                base_name = base_name[:-3]  # Remove .md extension
            pdf_path = f"{base_name}.pdf"

        # Create markdown content
        markdown_content = self._generate_markdown_content(include_images)

        # Create a temporary markdown file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
            temp_file.write(markdown_content)
            temp_markdown_path = temp_file.name

        try:
            # Add timeout for PDF generation to prevent hanging
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError("PDF generation timed out")

            # Set timeout to 60 seconds
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(60)

            try:
                self._convert_markdown_to_pdf(temp_markdown_path, pdf_path)
                print(f"Conversation history saved as PDF: {pdf_path}")
                print(f"Total steps recorded: {len(self.log)}")
            finally:
                signal.alarm(0)  # Cancel the alarm

        except TimeoutError:
            print("Warning: PDF generation timed out after 60 seconds")
        except Exception as e:
            print(f"Warning: Could not convert to PDF: {e}")
        finally:
            # Clean up the temporary markdown file
            try:
                os.unlink(temp_markdown_path)
            except OSError:
                pass  # File might already be deleted

    def _generate_markdown_content(self, include_images: bool = True) -> str:
        """Generate markdown content from conversation history using both log and conversation state.

        This function processes the agent's conversation history from either the conversation
        state (if available) or the internal log to create a formatted markdown document.
        It handles step numbering, message processing, and content formatting.

        Args:
            include_images: Whether to include captured plots and images in the output.
                          Defaults to True.

        Returns:
            Formatted markdown string containing the complete conversation history
            with proper step numbering and content structure.
        """

        # Initialize content and tracking variables
        content = """# Biomni Agent Conversation History

"""
        added_plots = set()
        step_number = 0
        first_human_shown = False

        # Get data source (conversation state or log)
        messages = self._get_messages_for_processing()

        # Process all messages using unified logic
        for message_data in messages:
            content, step_number, first_human_shown = self._process_message(
                message_data, content, step_number, first_human_shown, added_plots, include_images
            )

        return content

    def _get_messages_for_processing(self):
        """Get messages from conversation state or fallback to log.

        This function determines the best source for conversation messages, prioritizing
        the conversation state if available, otherwise falling back to the internal log.
        It normalizes the messages into a unified format for processing.

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        conversation_state = getattr(self, "_conversation_state", None)

        if conversation_state and hasattr(conversation_state, "get") and "messages" in conversation_state:
            print(f"DEBUG: Using conversation state with {len(conversation_state['messages'])} messages")
            return self._normalize_conversation_state_messages(conversation_state["messages"])
        else:
            print(f"DEBUG: Using self.log with {len(self.log)} entries")
            return self._normalize_log_messages(self.log)

    def _normalize_conversation_state_messages(self, messages):
        """Convert conversation state messages to unified format.

        This function takes LangChain message objects from the conversation state and
        converts them into a standardized dictionary format that the markdown generation
        system can work with. It extracts content and determines message types.

        Args:
            messages: List of LangChain message objects (HumanMessage, AIMessage, etc.)

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        normalized = []
        for message in messages:
            if hasattr(message, "content"):
                content = str(message.content)
            else:
                content = str(message)

            # Determine message type
            if isinstance(message, HumanMessage):
                msg_type = "human"
            elif isinstance(message, AIMessage):
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": message})

        return normalized

    def _normalize_log_messages(self, log_entries):
        """Convert log entries to unified format.

        This function takes internal log entries and converts them into the same
        standardized format as conversation state messages. It parses the log format
        to determine message types and extract content.

        Args:
            log_entries: List of log entry strings from the agent's internal log

        Returns:
            List of normalized message dictionaries with 'content', 'type', and 'original' keys
        """
        normalized = []
        for log_entry in log_entries:
            content = str(log_entry)

            # Determine message type from log format
            if "Human Message" in content:
                msg_type = "human"
            elif "Ai Message" in content:
                msg_type = "ai"
            else:
                msg_type = "other"

            normalized.append({"content": content, "type": msg_type, "original": log_entry})

        return normalized

    def _process_message(self, message_data, content, step_number, first_human_shown, added_plots, include_images):
        """Process a single message and return updated state.

        This function is the main dispatcher for processing individual messages in the
        conversation history. It determines the message type and delegates to the
        appropriate processing function.

        Args:
            message_data: Dictionary containing 'content', 'type', and 'original' keys
            content: Current markdown content string
            step_number: Current step number counter
            first_human_shown: Boolean flag indicating if first human message was shown
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_step_number, updated_first_human_shown)
        """
        clean_output = clean_message_content(message_data["content"])
        msg_type = message_data["type"]

        if msg_type == "human":
            return self._process_human_message(clean_output, content, step_number, first_human_shown)
        elif msg_type == "ai":
            return self._process_ai_message(clean_output, content, step_number, added_plots, include_images)
        else:
            return self._process_other_message(
                clean_output, content, step_number, first_human_shown, added_plots, include_images
            )

    def _process_human_message(self, clean_output, content, step_number, first_human_shown):
        """Process human messages.

        This function handles human messages in the conversation history. It identifies
        parsing error messages and displays them appropriately, or formats the first
        human prompt as a special section.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter (unchanged for human messages)
            first_human_shown: Boolean flag indicating if first human message was shown

        Returns:
            Tuple of (updated_content, step_number, updated_first_human_shown)

        Note:
            Human messages don't increment the step counter as they are not considered
            steps in the agent's process.
        """
        if "each response must include thinking process" in clean_output.lower():
            parsing_error_content = create_parsing_error_html()
            content += f"{parsing_error_content}\n\n"
        elif not first_human_shown:
            content += "#### Human Prompt\n\n"
            content += f"*{clean_output}*\n\n"
            first_human_shown = True

        return content, step_number, first_human_shown  # step_number unchanged

    def _process_ai_message(self, clean_output, content, step_number, added_plots, include_images):
        """Process AI messages.

        This function handles AI messages in the conversation history. It can process
        both regular AI responses and messages containing observation tags. It handles
        step numbering, execution results, and content formatting.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_step_number, True)

        Note:
            This function can split messages containing observation tags and process
            each part separately, with observations formatted as terminal blocks.
        """
        # Check if this message contains observation tags and process accordingly
        import re

        observation_pattern = r"<observation>(.*?)</observation>"
        observation_matches = re.findall(observation_pattern, clean_output, re.DOTALL | re.IGNORECASE)

        if observation_matches:
            # Extract content before, between, and after observation tags
            parts = re.split(observation_pattern, clean_output, flags=re.DOTALL | re.IGNORECASE)

            # Process each part
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Even indices are non-observation content
                    if part.strip():
                        # This is regular content - process it normally
                        if not should_skip_message(part):
                            if part.strip():
                                step_number += 1
                                content += f"#### Step {step_number}\n\n"

                                # Handle execution results if present
                                execution_results = getattr(self, "_execution_results", None)
                                if has_execution_results(part, execution_results):
                                    content, added_plots = self._process_execution_with_results(
                                        part, content, added_plots, include_images, execution_results
                                    )
                                else:
                                    content = self._process_regular_ai_message(part, content)
                else:  # Odd indices are observation content
                    if part.strip():
                        # This is observation content - format as terminal
                        formatted_observation = format_observation_as_terminal(f"<observation>{part}</observation>")
                        if formatted_observation is not None:
                            content += f"{formatted_observation}\n\n"

            return content, step_number, True

        # Skip empty or error messages
        if should_skip_message(clean_output):
            return content, step_number, True

        if clean_output.strip():
            step_number += 1
            content += f"#### Step {step_number}\n\n"

            # Handle execution results if present
            execution_results = getattr(self, "_execution_results", None)
            if has_execution_results(clean_output, execution_results):
                content, added_plots = self._process_execution_with_results(
                    clean_output, content, added_plots, include_images, execution_results
                )
            else:
                content = self._process_regular_ai_message(clean_output, content)

        return content, step_number, True

    def _process_other_message(
        self, clean_output, content, step_number, first_human_shown, added_plots, include_images
    ):
        """Process other message types.

        This function handles message types that are neither human nor AI messages.
        It checks for observation tags and processes them accordingly, or adds the
        content as regular text.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            step_number: Current step number counter
            first_human_shown: Boolean flag indicating if first human message was shown
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, step_number, first_human_shown)
        """
        # Check if this is actually an observation (has <observation> tags)
        import re

        if not re.search(r"<observation>", clean_output, re.IGNORECASE):
            content += f"{clean_output}\n\n"
        return content, step_number, first_human_shown

    def _process_execution_with_results(self, clean_output, content, added_plots, include_images, execution_results):
        """Process AI message with execution results.

        This function handles AI messages that have associated execution results.
        It finds the matching execution result and adds any captured plots or images
        to the content.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output
            execution_results: List of execution result dictionaries

        Returns:
            Tuple of (updated_content, updated_added_plots)
        """
        matching_execution = find_matching_execution(clean_output, execution_results)

        if matching_execution:
            content = self._format_and_add_content(clean_output, content)
            content, added_plots = self._add_execution_plots(matching_execution, content, added_plots, include_images)
        else:
            content = self._format_and_add_content(clean_output, content)

        return content, added_plots

    def _format_and_add_content(self, clean_output, content):
        """Format and add content to markdown.

        This function applies formatting to AI message content before adding it to the
        markdown. It processes lists, execute tags, and tool calls to create properly
        formatted markdown content.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string

        Returns:
            Updated markdown content string with formatted content added
        """
        # Process lists first, then execute tags
        formatted_content = format_lists_in_text(clean_output)

        # Create a wrapper function for the tool parsing
        def parse_tool_calls_wrapper(code):
            return self._parse_tool_calls_with_modules(code)

        formatted_content = format_execute_tags_in_content(formatted_content, parse_tool_calls_wrapper)
        return content + f"{formatted_content}\n\n"

    def _add_execution_plots(self, matching_execution, content, added_plots, include_images):
        """Add plots from execution results.

        This function adds captured plots and images from execution results to the
        markdown content. It prevents duplicate plots from being added multiple times.

        Args:
            matching_execution: Execution result dictionary containing image data
            content: Current markdown content string
            added_plots: Set of already added plot data to avoid duplicates
            include_images: Whether to include images in the output

        Returns:
            Tuple of (updated_content, updated_added_plots)
        """
        if include_images and matching_execution.get("images"):
            for plot_data in matching_execution["images"]:
                if plot_data not in added_plots:
                    content += f"![Plot]({plot_data})\n\n"
                    added_plots.add(plot_data)
        return content, added_plots

    def _process_regular_ai_message(self, clean_output, content):
        """Process regular AI message without execution results.

        This function handles AI messages that don't have associated execution results.
        It applies standard formatting and adds the content to the markdown.

        Args:
            clean_output: Cleaned message content with ANSI codes removed
            content: Current markdown content string

        Returns:
            Updated markdown content string with formatted content added
        """
        return self._format_and_add_content(clean_output, content)

    def _convert_markdown_to_pdf(self, markdown_path: str, pdf_path: str) -> None:
        """Convert markdown file to PDF using weasyprint or markdown2pdf.

        This function is a wrapper around the utility function for converting markdown
        to PDF. It provides a clean interface for the agent to convert conversation
        history to PDF format.

        Args:
            markdown_path: Path to the input markdown file
            pdf_path: Path where the output PDF file should be saved

        Note:
            This function delegates to the convert_markdown_to_pdf utility function
            which handles multiple PDF conversion libraries and fallbacks.
        """
        convert_markdown_to_pdf(markdown_path, pdf_path)

    def _clear_execution_plots(self):
        """Clear execution plots before new execution.

        This function clears any previously captured plots from the execution environment
        before starting a new execution. This prevents old plots from appearing in
        new execution results.

        Note:
            This function calls the clear_captured_plots utility function and handles
            any exceptions gracefully to prevent execution failures.
        """
        try:
            from biomni.tool.support_tools import clear_captured_plots

            clear_captured_plots()
        except Exception as e:
            print(f"Warning: Could not clear execution plots: {e}")

    def _generate_mcp_wrapper_from_biomni_schema(self, original_func, func_name, required_params, optional_params):
        """Generate wrapper function based on Biomni schema format."""
        import inspect

        # Combine all parameters
        all_params = required_params + optional_params

        if not all_params:
            # No parameters
            def wrapper() -> dict:
                try:
                    result = original_func()
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__
            return wrapper

        else:
            # Has parameters
            def wrapper(**kwargs) -> dict:
                try:
                    # Build arguments dict
                    filtered_kwargs = {}

                    # Add required parameters
                    for param_info in required_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    # Add optional parameters only if provided and not None
                    for param_info in optional_params:
                        param_name = param_info["name"]
                        if param_name in kwargs and kwargs[param_name] is not None:
                            filtered_kwargs[param_name] = kwargs[param_name]

                    result = original_func(**filtered_kwargs)
                    if isinstance(result, dict):
                        return result
                    return {"result": result}
                except Exception as e:
                    return {"error": str(e)}

            # Set function metadata
            wrapper.__name__ = func_name
            wrapper.__doc__ = original_func.__doc__

            # Create proper signature
            new_params = []

            # Map your types to Python types
            type_map = {"str": str, "int": int, "float": float, "bool": bool, "List[str]": list[str], "dict": dict}

            # Add required parameters
            for param_info in required_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                new_params.append(inspect.Parameter(param_name, inspect.Parameter.KEYWORD_ONLY, annotation=param_type))

            # Add optional parameters
            for param_info in optional_params:
                param_name = param_info["name"]
                param_type_str = param_info["type"]
                param_type = type_map.get(param_type_str, str)

                # Make it optional
                optional_type = param_type | None

                new_params.append(
                    inspect.Parameter(
                        param_name, inspect.Parameter.KEYWORD_ONLY, default=None, annotation=optional_type
                    )
                )

            # Set the signature
            wrapper.__signature__ = inspect.Signature(new_params, return_annotation=dict)

            return wrapper

    def _launch_gradio_demo_premium(self, thread_id=42, share=False, server_name="0.0.0.0", require_verification=False):
        """Launch the premium Gradio workbench UI for the A1 agent."""
        try:
            import gradio as gr
            from gradio import ChatMessage
        except ImportError:
            raise ImportError("Gradio is not installed. Please install it with: pip install gradio") from None

        from time import time

        supported_extensions = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf")
        working_message = _gradio_working_message()

        self.main_history_copy = []
        available_access_codes = ["Biomni2025"]

        def verify_access_code(code):
            if code in available_access_codes:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(value="Incorrect access code. Please check your access code.", visible=True),
            )

        def generate_response(prompt_input, inner_history=None, main_history=None):
            if main_history is None:
                main_history = []
            if inner_history is None:
                inner_history = []

            def surface_main_status(content: str, title: str, log: str):
                status_message = ChatMessage(role="assistant", content=content, metadata={"title": title, "log": log})
                if main_history and getattr(main_history[-1], "content", None) == working_message:
                    main_history[-1] = status_message
                else:
                    main_history.append(status_message)
                self.main_history_copy += [{"role": "assistant", "content": content}]

            try:
                text_input = prompt_input.get("text", "")
                files = prompt_input.get("files", [])

                self.main_history_copy += [{"role": "user", "content": text_input}]
                main_history.append(ChatMessage(role="user", content=text_input if text_input else "[Uploaded file]"))
                main_history.append(
                    ChatMessage(
                        role="assistant",
                        content=working_message,
                        metadata={"title": "Running", "log": "Biomni accepted the task and is waiting for progress."},
                    )
                )
                yield inner_history, main_history

                for file_info in files:
                    file_path = file_info
                    text_input += f"\n\nUser uploaded this file: {file_path}\nPlease use it if needed."

                agent_messages = []
                for msg in self.main_history_copy:
                    if msg["role"] == "user":
                        agent_messages.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant" and msg["content"] != working_message:
                        agent_messages.append(AIMessage(content=msg["content"]))

                agent_messages.append(HumanMessage(content=text_input))

                inputs = {"messages": agent_messages, "next_step": None}
                config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}
                t = time()
                solution_found = False
                final_message = ""
                stream_timeout_seconds = _gradio_stream_timeout_seconds(self.timeout_seconds)

                if self.use_tool_retriever:
                    print("Using tool retriever...")
                    inner_history.append(
                        ChatMessage(
                            role="assistant",
                            content="Selecting relevant tools, libraries, and optional data assets for this run.",
                            metadata={"title": "Resource Selection"},
                        )
                    )
                    yield inner_history, main_history

                    try:
                        selected_resources_names = self._prepare_resources_for_retrieval(text_input)
                        if selected_resources_names:
                            self.update_system_prompt_with_selected_resources(selected_resources_names)
                    except Exception as e:
                        print(f"Warning: Tool retrieval failed: {e}")
                        print("Continuing without tool retrieval...")
                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content="Tool retrieval is unavailable for this run, so Biomni will continue with the base toolset.",
                                metadata={"title": "Resource Selection"},
                            )
                        )
                        yield inner_history, main_history

                code_execution_messages = []
                stream_queue: queue.Queue[tuple[str, Any]] = queue.Queue()

                def pump_stream():
                    try:
                        for event in self.app.stream(inputs, stream_mode="values", config=config):
                            stream_queue.put(("event", event))
                        stream_queue.put(("done", None))
                    except Exception as exc:
                        stream_queue.put(("error", exc))

                threading.Thread(target=pump_stream, daemon=True).start()

                while True:
                    try:
                        stream_kind, payload = stream_queue.get(timeout=stream_timeout_seconds)
                    except queue.Empty:
                        timeout_message = _gradio_timeout_message(stream_timeout_seconds)
                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content=timeout_message,
                                metadata={"title": "Run Timed Out", "log": "No streamed events arrived before timeout."},
                            )
                        )
                        surface_main_status(timeout_message, "Run Timed Out", "The run stopped after an inactivity timeout.")
                        yield inner_history, main_history
                        return

                    if stream_kind == "error":
                        failure_message = _gradio_failure_message(payload)
                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content=failure_message,
                                metadata={"title": "Run Failed", "log": "An exception escaped the streaming loop."},
                            )
                        )
                        surface_main_status(failure_message, "Run Failed", "The backend raised an exception during the run.")
                        yield inner_history, main_history
                        return

                    if stream_kind == "done":
                        break

                    s = payload
                    t_step = time() - t
                    message = s["messages"][-1]
                    if isinstance(message.content, str):
                        final_message = message.content

                    if message.content == text_input:
                        t = time()
                        continue

                    if isinstance(message.content, str):
                        tag_positions = []
                        for tag in ["<execute>", "<solution>", "<observation>"]:
                            pos = message.content.find(tag)
                            if pos != -1:
                                tag_positions.append(pos)

                        if tag_positions:
                            first_tag_pos = min(tag_positions)
                            thinking = message.content[:first_tag_pos].strip()
                            if thinking:
                                inner_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content=thinking,
                                        metadata={"title": "Reasoning", "log": "Agent reasoning trace"},
                                    )
                                )
                                yield inner_history, main_history

                        solution_match = re.search(r"<solution>(.*?)</solution>", message.content, re.DOTALL)
                        if solution_match and not solution_found:
                            solution = solution_match.group(1).strip()
                            main_history.append(
                                ChatMessage(
                                    role="assistant",
                                    content=solution,
                                    metadata={"title": "Final Answer", "log": "Final answer provided by the agent"},
                                )
                            )
                            self.main_history_copy += [{"role": "assistant", "content": solution}]
                            solution_found = True
                            yield inner_history, main_history

                        execute_match = re.search(r"<execute>(.*?)</execute>", message.content, re.DOTALL)
                        if execute_match:
                            code = execute_match.group(1).strip()
                            language = "python"
                            if code.strip().startswith("#!R"):
                                language = "r"
                                code = re.sub(r"^#!R", "", code, count=1).strip()
                            elif code.strip().startswith("#!BASH") or code.strip().startswith("#!CLI"):
                                language = "bash"
                                code = re.sub(r"^#!BASH|^#!CLI", "", code, count=1).strip()

                            code_msg = ChatMessage(
                                role="assistant",
                                content=f"##### Code\n```{language}\n{code}\n```",
                                metadata={
                                    "title": "Executing Code",
                                    "log": f"Executing {language.capitalize()} code block...",
                                    "status": "pending",
                                    "start_time": t,
                                },
                            )
                            inner_history.append(code_msg)
                            code_execution_messages.append(code_msg)
                            yield inner_history, main_history

                        observation_match = re.search(r"<observation>(.*?)</observation>", message.content, re.DOTALL)
                        if observation_match:
                            observation = observation_match.group(1).strip()

                            if code_execution_messages:
                                code_msg = code_execution_messages[-1]
                                code_msg.metadata.update(
                                    {
                                        "status": "done",
                                        "duration": t_step,
                                        "log": f"Code execution completed in {t_step:.2f}s",
                                    }
                                )

                            inner_history.append(
                                ChatMessage(
                                    role="assistant",
                                    content=f"##### Observation\n```\n{observation}\n```",
                                    metadata={
                                        "status": "done",
                                        "duration": t_step,
                                        "log": "Observation from code execution",
                                        "collapsed": True,
                                        "collapsible": True,
                                        "title": "Execution Output",
                                    },
                                )
                            )
                            yield inner_history, main_history

                            if isinstance(observation, str) and any(ext in observation for ext in supported_extensions):
                                matches = re.findall(r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf))", observation)
                                valid_matches = []
                                for match in matches:
                                    if not (
                                        match.startswith("Warning:") or match.startswith("Error:") or match.startswith("'")
                                    ) and not match.startswith("."):
                                        valid_matches.append(match)

                                if valid_matches:
                                    inner_history.append(
                                        ChatMessage(
                                            role="assistant",
                                            content="Generated files are available below.",
                                            metadata={"title": "Generated Files", "log": "Files generated by the agent"},
                                        )
                                    )

                                    for file_path in valid_matches:
                                        file_path = file_path.strip("\"'").strip()
                                        abs_path = None

                                        if os.path.isabs(file_path) and os.path.exists(file_path):
                                            abs_path = file_path
                                        elif os.path.exists(os.path.join(os.getcwd(), file_path)):
                                            abs_path = os.path.join(os.getcwd(), file_path)
                                        elif (
                                            hasattr(self, "path")
                                            and self.path
                                            and os.path.exists(os.path.join(self.path, file_path))
                                        ):
                                            abs_path = os.path.join(self.path, file_path)

                                        if abs_path:
                                            if file_path.lower().endswith(".pdf"):
                                                inner_history.append(
                                                    ChatMessage(
                                                        role="assistant",
                                                        content=f"PDF ready at: {abs_path}",
                                                        metadata={"title": "PDF Preview"},
                                                    )
                                                )
                                            else:
                                                inner_history.append(
                                                    ChatMessage(
                                                        role="assistant",
                                                        content=gr.Image(abs_path),
                                                        metadata={"title": "Image Preview"},
                                                    )
                                                )

                                    yield inner_history, main_history

                    t = time()

                if not solution_found:
                    solution_match = re.search(r"<solution>(.*?)</solution>", final_message, re.DOTALL)
                    if solution_match:
                        solution = solution_match.group(1).strip()
                        main_history.append(ChatMessage(role="assistant", content=solution, metadata={"title": "Final Answer"}))
                        self.main_history_copy += [{"role": "assistant", "content": solution}]
                    else:
                        cleaned_content = re.sub(r"<execute>.*?</execute>", "", final_message, flags=re.DOTALL)
                        cleaned_content = re.sub(r"<observation>.*?</observation>", "", cleaned_content, flags=re.DOTALL)
                        cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

                        if cleaned_content.strip():
                            main_history.append(
                                ChatMessage(role="assistant", content=cleaned_content.strip(), metadata={"title": "Summary"})
                            )
                            self.main_history_copy += [{"role": "assistant", "content": cleaned_content.strip()}]
                        else:
                            main_history.append(
                                ChatMessage(
                                    role="assistant",
                                    content="Task completed. Review the execution trace for full detail.",
                                    metadata={"title": "Summary"},
                                )
                            )
                            self.main_history_copy += [{"role": "assistant", "content": "Task completed."}]

                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content="Returning the result to the main research console.",
                        metadata={"title": "Complete"},
                    )
                )
                yield inner_history, main_history
            except Exception as e:
                failure_message = _gradio_failure_message(e)
                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content=failure_message,
                        metadata={"title": "Run Failed", "log": "The Gradio response generator raised an exception."},
                    )
                )
                surface_main_status(
                    failure_message,
                    "Run Failed",
                    "The Gradio response generator raised an exception before the run could finish.",
                )
                yield inner_history, main_history

        def like(data: gr.LikeData):
            print("User liked the response")
            print(f"Index: {data.index}, Liked: {data.liked}")

        def render_status_html(locale: str = "en") -> str:
            copy = _gradio_locale_copy(locale)
            runtime_model = self.llm_model_name or getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None)
            runtime_model = runtime_model or default_config.llm
            runtime_provider = _gradio_match_llm_provider(str(runtime_model), self.llm_base_url)
            mode_value = copy["status_academic"] if not self.commercial_mode else copy["status_commercial"]
            retriever_value = copy["status_enabled"] if self.use_tool_retriever else copy["status_disabled"]
            data_lake_value = copy["status_data_lake_loaded"] if data_lake_state == "Loaded" else copy["status_data_lake_on_demand"]
            return "".join(
                [
                    _gradio_status_pill_html(copy["status_provider"], runtime_provider),
                    _gradio_status_pill_html(copy["status_model"], str(runtime_model)),
                    _gradio_status_pill_html(copy["status_mode"], mode_value),
                    _gradio_status_pill_html(copy["status_data_lake"], data_lake_value),
                    _gradio_status_pill_html(copy["status_retriever"], retriever_value),
                    _gradio_status_pill_html(copy["status_status"], copy["status_ready"]),
                ]
            )

        def apply_runtime_llm_settings(
            provider_name: str,
            model_value: str,
            base_url_value: str,
            api_key_value: str,
            locale: str,
        ):
            copy = _gradio_locale_copy(locale)
            try:
                applied = self.update_runtime_llm_config(
                    provider_name=provider_name,
                    model=model_value,
                    base_url=base_url_value,
                    api_key=api_key_value,
                )
                feedback = copy["apply_success"].format(
                    provider=escape(applied["provider"]),
                    model=escape(applied["model"]),
                    base_url=escape(applied["base_url"]),
                )
            except Exception as exc:
                feedback = copy["apply_failure"].format(error=escape(str(exc)))
            return render_status_html(locale), feedback, ""

        def populate_provider_defaults(provider_name: str, current_model_value: str, current_base_url_value: str):
            defaults = _gradio_llm_provider_form_state(
                provider_name,
                current_model=current_model_value,
                current_base_url=current_base_url_value,
            )
            return (
                gr.Dropdown(
                    choices=defaults["model_choices"],
                    value=defaults["model"],
                    allow_custom_value=True,
                ),
                defaults["base_url"],
            )

        def sync_model_routing_form(provider_name: str, model_value: str, base_url_value: str):
            defaults = _gradio_llm_provider_form_state(
                provider_name,
                current_model=model_value,
                current_base_url=base_url_value,
            )
            return (
                gr.Dropdown(
                    choices=list(_gradio_llm_provider_presets().keys()),
                    value=defaults["provider"],
                ),
                gr.Dropdown(
                    choices=defaults["model_choices"],
                    value=defaults["model"],
                    allow_custom_value=True,
                ),
                defaults["base_url"],
            )

        def render_locale_ui(
            locale: str,
            current_provider: str,
            current_model_value: str,
            current_base_url_value: str,
        ):
            copy = _gradio_locale_copy(locale)
            defaults = _gradio_llm_provider_form_state(
                current_provider,
                current_model=current_model_value,
                current_base_url=current_base_url_value,
            )
            return (
                gr.Dropdown(label=copy["language_label"], choices=copy["language_choices"], value=locale),
                _gradio_header_html(locale),
                copy["verification_copy"],
                gr.Textbox(label=copy["verification_label"], type="password"),
                gr.Button(copy["verification_button"]),
                f"<div id='biomni-status-row'>{render_status_html(locale)}</div>",
                _gradio_quickstart_html(locale),
                _gradio_run_guide_html(locale),
                _gradio_panel_header_html(
                    copy["panel_primary_kicker"],
                    copy["panel_primary_title"],
                    copy["panel_primary_copy"],
                ),
                gr.Chatbot(label=copy["panel_primary_title"]),
                _gradio_panel_header_html(
                    copy["panel_operations_kicker"],
                    copy["panel_operations_title"],
                    copy["panel_operations_copy"],
                ),
                gr.Chatbot(label=copy["panel_operations_title"]),
                _gradio_composer_html(locale),
                gr.Accordion(label=copy["model_routing_title"]),
                f"<p class='biomni-llm-note'>{escape(copy['model_routing_copy'])}</p>",
                gr.Dropdown(label=copy["provider_preset_label"]),
                gr.Dropdown(
                    label=copy["model_label"],
                    choices=defaults["model_choices"],
                    value=defaults["model"],
                    allow_custom_value=True,
                    info=copy["model_info"],
                ),
                gr.Textbox(
                    label=copy["base_url_label"],
                    value=defaults["base_url"],
                    placeholder="https://open.bigmodel.cn/api/paas/v4/",
                ),
                gr.Textbox(
                    label=copy["api_key_label"],
                    type="password",
                    placeholder=copy["api_key_placeholder"],
                ),
                gr.Button(copy["apply_model_routing"]),
                gr.Button(copy["quick_action_drug"]),
                gr.Button(copy["quick_action_literature"]),
                gr.Button(copy["quick_action_single_cell"]),
                gr.Button(copy["quick_action_clear"]),
                gr.MultimodalTextbox(placeholder=copy["prompt_placeholder"]),
            )

        model_name = getattr(self.llm, "model_name", None) or getattr(self.llm, "model", None) or default_config.llm
        runtime_provider = _gradio_match_llm_provider(str(model_name), self.llm_base_url)
        runtime_defaults = _gradio_llm_provider_form_state(
            runtime_provider,
            current_model=str(model_name),
            current_base_url=self.llm_base_url or "",
        )
        data_lake_dir = os.path.join(self.path, "data_lake") if hasattr(self, "path") else ""
        data_lake_state = "Loaded" if data_lake_dir and os.path.isdir(data_lake_dir) and os.listdir(data_lake_dir) else "On demand"
        default_locale = "zh"
        default_copy = _gradio_locale_copy(default_locale)
        status_html = render_status_html(default_locale)

        with gr.Blocks(css=_gradio_workbench_css(), title="Biomni Research Workbench", fill_height=True) as demo:
            verification_container = gr.Group(
                visible=require_verification,
                elem_id="biomni-verify",
                elem_classes=["biomni-panel"],
            )
            main_interface_container = gr.Group(visible=not require_verification)
            locale_state = gr.State(default_locale)

            with verification_container:
                verification_header = gr.HTML(_gradio_header_html(default_locale))
                verification_copy = gr.Markdown(default_copy["verification_copy"])
                access_code_input = gr.Textbox(label=default_copy["verification_label"], type="password")
                access_error_msg = gr.Markdown(visible=False)
                verify_btn = gr.Button(default_copy["verification_button"])
                verify_btn.click(
                    fn=verify_access_code,
                    inputs=[access_code_input],
                    outputs=[verification_container, main_interface_container, access_error_msg],
                )

            with main_interface_container:
                with gr.Column(elem_id="biomni-shell"):
                    with gr.Group(elem_id="biomni-header"):
                        header_html = gr.HTML(_gradio_header_html(default_locale))
                        language_selector = gr.Dropdown(
                            label=default_copy["language_label"],
                            choices=default_copy["language_choices"],
                            value=default_locale,
                            scale=0,
                            elem_id="biomni-language-selector",
                        )
                        status_row = gr.HTML(f"<div id='biomni-status-row'>{status_html}</div>")
                    with gr.Row(elem_id="biomni-hero-row", equal_height=True):
                        with gr.Column(scale=6, elem_classes=["biomni-panel"]):
                            quickstart_html = gr.HTML(_gradio_quickstart_html(default_locale))
                        with gr.Column(scale=5, elem_classes=["biomni-panel"]):
                            run_guide_html = gr.HTML(_gradio_run_guide_html(default_locale))

                    with gr.Row(elem_id="biomni-workbench", equal_height=True):
                        with gr.Column(scale=6, elem_classes=["biomni-panel"]):
                            primary_panel_header = gr.HTML(
                                _gradio_panel_header_html(
                                    default_copy["panel_primary_kicker"],
                                    default_copy["panel_primary_title"],
                                    default_copy["panel_primary_copy"],
                                )
                            )
                            main_chatbot = gr.Chatbot(
                                label=default_copy["panel_primary_title"],
                                type="messages",
                                height=820,
                                show_copy_button=True,
                                show_share_button=True,
                                elem_id="biomni-main-chat",
                            )
                        with gr.Column(scale=5, elem_classes=["biomni-panel"]):
                            operations_panel_header = gr.HTML(
                                _gradio_panel_header_html(
                                    default_copy["panel_operations_kicker"],
                                    default_copy["panel_operations_title"],
                                    default_copy["panel_operations_copy"],
                                )
                            )
                            innerloop_chatbot = gr.Chatbot(
                                label=default_copy["panel_operations_title"],
                                type="messages",
                                height=820,
                                show_copy_button=True,
                                show_share_button=True,
                                elem_id="biomni-trace-chat",
                            )

                    with gr.Group(elem_id="biomni-composer", elem_classes=["biomni-panel"]):
                        composer_html = gr.HTML(_gradio_composer_html(default_locale))
                        with gr.Accordion(default_copy["model_routing_title"], open=False, elem_id="biomni-llm-routing") as llm_routing_accordion:
                            llm_routing_note = gr.Markdown(
                                f"<p class='biomni-llm-note'>{escape(default_copy['model_routing_copy'])}</p>"
                            )
                            with gr.Row():
                                provider_selector = gr.Dropdown(
                                    choices=list(_gradio_llm_provider_presets().keys()),
                                    value=runtime_defaults["provider"],
                                    label=default_copy["provider_preset_label"],
                                )
                                model_selector = gr.Dropdown(
                                    label=default_copy["model_label"],
                                    choices=runtime_defaults["model_choices"],
                                    value=runtime_defaults["model"],
                                    allow_custom_value=True,
                                    info=default_copy["model_info"],
                                )
                            with gr.Row():
                                base_url_selector = gr.Textbox(
                                    label=default_copy["base_url_label"],
                                    value=runtime_defaults["base_url"],
                                    placeholder="https://open.bigmodel.cn/api/paas/v4/",
                                )
                                api_key_selector = gr.Textbox(
                                    label=default_copy["api_key_label"],
                                    type="password",
                                    placeholder=default_copy["api_key_placeholder"],
                                )
                            apply_llm_button = gr.Button(default_copy["apply_model_routing"], variant="secondary")
                            llm_feedback = gr.Markdown("", elem_id="biomni-llm-feedback")
                        with gr.Row(elem_id="biomni-quick-actions"):
                            drug_label_btn = gr.Button(
                                default_copy["quick_action_drug"],
                                variant="secondary",
                                size="sm",
                                elem_classes=["biomni-quick-btn"],
                            )
                            literature_btn = gr.Button(
                                default_copy["quick_action_literature"],
                                variant="secondary",
                                size="sm",
                                elem_classes=["biomni-quick-btn"],
                            )
                            sc_btn = gr.Button(
                                default_copy["quick_action_single_cell"],
                                variant="secondary",
                                size="sm",
                                elem_classes=["biomni-quick-btn"],
                            )
                            clear_btn = gr.Button(
                                default_copy["quick_action_clear"],
                                variant="secondary",
                                size="sm",
                                elem_classes=["biomni-quick-btn"],
                            )
                        prompt_input = gr.MultimodalTextbox(
                            interactive=True,
                            file_count="multiple",
                            placeholder=default_copy["prompt_placeholder"],
                            show_label=False,
                            elem_id="biomni-prompt",
                        )

                language_selector.change(
                    render_locale_ui,
                    [language_selector, provider_selector, model_selector, base_url_selector],
                    [
                        language_selector,
                        header_html,
                        verification_copy,
                        access_code_input,
                        verify_btn,
                        status_row,
                        quickstart_html,
                        run_guide_html,
                        primary_panel_header,
                        main_chatbot,
                        operations_panel_header,
                        innerloop_chatbot,
                        composer_html,
                        llm_routing_accordion,
                        llm_routing_note,
                        provider_selector,
                        model_selector,
                        base_url_selector,
                        api_key_selector,
                        apply_llm_button,
                        drug_label_btn,
                        literature_btn,
                        sc_btn,
                        clear_btn,
                        prompt_input,
                    ],
                    queue=False,
                    show_progress="hidden",
                ).then(
                    lambda locale: locale,
                    [language_selector],
                    [locale_state],
                    queue=False,
                    show_progress="hidden",
                )
                provider_selector.change(
                    populate_provider_defaults,
                    [provider_selector, model_selector, base_url_selector],
                    [model_selector, base_url_selector],
                    queue=False,
                    show_progress="hidden",
                )
                model_selector.change(
                    sync_model_routing_form,
                    [provider_selector, model_selector, base_url_selector],
                    [provider_selector, model_selector, base_url_selector],
                    queue=False,
                    show_progress="hidden",
                )
                apply_llm_button.click(
                    apply_runtime_llm_settings,
                    [provider_selector, model_selector, base_url_selector, api_key_selector, locale_state],
                    [status_row, llm_feedback, api_key_selector],
                    queue=False,
                )
                drug_label_btn.click(
                    lambda locale: _gradio_prompt_payload("Drug Label Review", locale),
                    [locale_state],
                    [prompt_input],
                    queue=False,
                    show_progress="hidden",
                )
                literature_btn.click(
                    lambda locale: _gradio_prompt_payload("Literature Triage", locale),
                    [locale_state],
                    [prompt_input],
                    queue=False,
                    show_progress="hidden",
                )
                sc_btn.click(
                    lambda locale: _gradio_prompt_payload("Single-Cell Readout", locale),
                    [locale_state],
                    [prompt_input],
                    queue=False,
                    show_progress="hidden",
                )
                clear_btn.click(
                    _gradio_empty_prompt_payload,
                    None,
                    [prompt_input],
                    queue=False,
                    show_progress="hidden",
                )
                prompt_input.submit(
                    generate_response,
                    [prompt_input, innerloop_chatbot, main_chatbot],
                    [innerloop_chatbot, main_chatbot],
                ).then(
                    _gradio_empty_prompt_payload,
                    None,
                    [prompt_input],
                )
                main_chatbot.like(like)

        print(f"Launching Gradio demo on {server_name}:7860")
        demo.launch(share=share, server_name=server_name)

    def launch_gradio_demo(self, thread_id=42, share=False, server_name="0.0.0.0", require_verification=False):
        """Launch a full-featured Gradio UI for the A1 agent (adapted from codeact_copilot).

        Args:
            thread_id: Thread ID for the conversation
            share: Whether to create a public shareable link
            server_name: Server name/IP to bind to (default: "0.0.0.0")
            require_verification: If True, requires access code verification

        Example:
            >>> agent = A1()
            >>> agent.launch_gradio_demo()
        """
        return self._launch_gradio_demo_premium(
            thread_id=thread_id,
            share=share,
            server_name=server_name,
            require_verification=require_verification,
        )

        try:
            import gradio as gr
            from gradio import ChatMessage
        except ImportError:
            raise ImportError("Gradio is not installed. Please install it with: pip install gradio") from None

        import os
        from time import time

        # Define supported file extensions
        SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf")

        self.main_history_copy = []

        # Available access codes (if verification is required)
        available_access_codes = ["Biomni2025"]

        # Function for verification page
        def verify_access_code(code):
            if code in available_access_codes:
                return gr.update(visible=False), gr.update(visible=True), gr.update(visible=False)
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    gr.update(value="Incorrect access code. Please check your access code.", visible=True),
                )

        def generate_response(prompt_input, inner_history=None, main_history=None):
            if main_history is None:
                main_history = []
            if inner_history is None:
                inner_history = []
            text_input = prompt_input.get("text", "")
            files = prompt_input.get("files", [])

            self.main_history_copy += [{"role": "user", "content": text_input}]
            main_history.append(ChatMessage(role="user", content=text_input if text_input else "[Uploaded file]"))

            # Add "Executor is working on it" message
            main_history.append(ChatMessage(role="assistant", content="Executor is working on it 👉"))
            yield inner_history, main_history

            # Process uploaded files if any
            for file_info in files:
                file_path = file_info
                text_input += f"\n\n User uploaded this file: {file_path}\n Please use it if needed."

            agent_messages = []
            for msg in self.main_history_copy:
                if msg["role"] == "user":
                    agent_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    if msg["content"] not in ["Executor is working on it 👉"]:
                        agent_messages.append(AIMessage(content=msg["content"]))

            agent_messages.append(HumanMessage(content=text_input))

            # Prepare inputs for the agent
            inputs = {"messages": agent_messages, "next_step": None}
            config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}

            # Stream the agent's responses
            t = time()
            solution_found = False

            # Configure the agent with tool retrieval if needed
            if self.use_tool_retriever:
                print("Using tool retriever...")
                inner_history.append(
                    ChatMessage(
                        role="assistant",
                        content="Retrieving relevant tools, data lake items, and libraries...",
                    )
                )
                yield inner_history, main_history

                try:
                    selected_resources_names = self._prepare_resources_for_retrieval(text_input)
                    if selected_resources_names:
                        self.update_system_prompt_with_selected_resources(selected_resources_names)
                except Exception as e:
                    print(f"Warning: Tool retrieval failed: {e}")
                    print("Continuing without tool retrieval...")
                    inner_history.append(
                        ChatMessage(
                            role="assistant",
                            content="Tool retrieval unavailable, proceeding with all tools...",
                        )
                    )
                    yield inner_history, main_history

            # Keep track of code execution messages
            code_execution_messages = []

            # Stream the agent's responses
            for s in self.app.stream(inputs, stream_mode="values", config=config):
                t_step = time() - t
                message = s["messages"][-1]

                # Skip the first message which is the input task
                if message.content == text_input:
                    t = time()
                    continue

                # Process the message
                if isinstance(message.content, str):
                    # Extract thinking/reasoning part (text before any tags)
                    tag_positions = []
                    for tag in ["<execute>", "<solution>", "<observation>"]:
                        pos = message.content.find(tag)
                        if pos != -1:
                            tag_positions.append(pos)

                    # If there are tags, extract the text before the first tag
                    if tag_positions:
                        first_tag_pos = min(tag_positions)
                        thinking = message.content[:first_tag_pos].strip()
                        if thinking:
                            inner_history.append(
                                ChatMessage(
                                    role="assistant",
                                    content=f"{thinking}",
                                    metadata={"title": "🤔 Reasoning", "log": "Agent's thinking process"},
                                )
                            )
                            yield inner_history, main_history

                    # Check for solution tag
                    solution_match = re.search(r"<solution>(.*?)</solution>", message.content, re.DOTALL)
                    if solution_match and not solution_found:
                        solution = solution_match.group(1).strip()
                        main_history.append(
                            ChatMessage(
                                role="assistant",
                                content=solution,
                                metadata={"title": "✅ Answer", "log": "Final answer provided by the agent"},
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": solution}]
                        solution_found = True
                        yield inner_history, main_history

                    # Check for execute tag
                    execute_match = re.search(r"<execute>(.*?)</execute>", message.content, re.DOTALL)
                    if execute_match:
                        code = execute_match.group(1).strip()
                        language = "python"
                        if code.strip().startswith("#!R"):
                            language = "r"
                            code = re.sub(r"^#!R", "", code, count=1).strip()
                        elif code.strip().startswith("#!BASH") or code.strip().startswith("#!CLI"):
                            language = "bash"
                            code = re.sub(r"^#!BASH|^#!CLI", "", code, count=1).strip()

                        code_msg = ChatMessage(
                            role="assistant",
                            content=f"##### Code: \n```{language}\n{code}\n```",
                            metadata={
                                "title": "🛠️ Executing code...",
                                "log": f"Executing {language.capitalize()} code block...",
                                "status": "pending",
                                "start_time": t,
                            },
                        )
                        inner_history.append(code_msg)
                        code_execution_messages.append(code_msg)
                        yield inner_history, main_history

                    # Check for observation
                    observation_match = re.search(r"<observation>(.*?)</observation>", message.content, re.DOTALL)
                    if observation_match:
                        observation = observation_match.group(1).strip()

                        # Update the status of the most recent code execution message
                        if code_execution_messages:
                            code_msg = code_execution_messages[-1]
                            code_msg.metadata.update(
                                {
                                    "status": "done",
                                    "duration": t_step,
                                    "log": f"Code execution completed in {t_step:.2f}s",
                                }
                            )

                        # Create a new message for the observation
                        inner_history.append(
                            ChatMessage(
                                role="assistant",
                                content=f"##### Observation: \n```\n{observation}\n```",
                                metadata={
                                    "status": "done",
                                    "duration": t_step,
                                    "log": "Observation from code execution",
                                    "collapsed": True,
                                    "collapsible": True,
                                },
                            )
                        )
                        yield inner_history, main_history

                        # Check for file paths in the observation
                        if isinstance(observation, str) and any(ext in observation for ext in SUPPORTED_EXTENSIONS):
                            matches = re.findall(r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf))", observation)

                            valid_matches = []
                            for match in matches:
                                if not (
                                    match.startswith("Warning:") or match.startswith("Error:") or match.startswith("'")
                                ):
                                    if not match.startswith("."):
                                        valid_matches.append(match)

                            if valid_matches:
                                inner_history.append(
                                    ChatMessage(
                                        role="assistant",
                                        content="",
                                        metadata={"title": "📁 Files", "log": "Files generated by the agent"},
                                    )
                                )

                                for file_path in valid_matches:
                                    file_path = file_path.strip("\"'").strip()

                                    abs_path = None
                                    if os.path.isabs(file_path) and os.path.exists(file_path):
                                        abs_path = file_path
                                    elif os.path.exists(os.path.join(os.getcwd(), file_path)):
                                        abs_path = os.path.join(os.getcwd(), file_path)
                                    elif (
                                        hasattr(self, "path")
                                        and self.path
                                        and os.path.exists(os.path.join(self.path, file_path))
                                    ):
                                        abs_path = os.path.join(self.path, file_path)

                                    if abs_path:
                                        if file_path.lower().endswith(".pdf"):
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=f"Found PDF at: {abs_path}",
                                                    metadata={"title": "📄 PDF File"},
                                                )
                                            )
                                        else:
                                            inner_history.append(
                                                ChatMessage(
                                                    role="assistant",
                                                    content=gr.Image(abs_path),
                                                    metadata={"title": "🖼️ Image Preview"},
                                                )
                                            )

                                yield inner_history, main_history

                t = time()

            # If no solution was found, add the final message
            if not solution_found:
                final_message = s["messages"][-1].content if s["messages"] else ""
                solution_match = re.search(r"<solution>(.*?)</solution>", final_message, re.DOTALL)
                if solution_match:
                    solution = solution_match.group(1).strip()
                    main_history.append(
                        ChatMessage(role="assistant", content=solution, metadata={"title": "✅ Solution"})
                    )
                    self.main_history_copy += [{"role": "assistant", "content": solution}]
                else:
                    cleaned_content = re.sub(r"<execute>.*?</execute>", "", final_message, flags=re.DOTALL)
                    cleaned_content = re.sub(r"<observation>.*?</observation>", "", cleaned_content, flags=re.DOTALL)
                    cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)

                    if cleaned_content.strip():
                        main_history.append(
                            ChatMessage(
                                role="assistant", content=cleaned_content.strip(), metadata={"title": "📝 Summary"}
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": cleaned_content.strip()}]
                    else:
                        main_history.append(
                            ChatMessage(
                                role="assistant",
                                content="Task completed. Please check the execution log for details.",
                                metadata={"title": "📝 Summary"},
                            )
                        )
                        self.main_history_copy += [{"role": "assistant", "content": "Task completed."}]

            # Add completion message
            inner_history.append(
                ChatMessage(
                    role="assistant",
                    content="👈 Returning the result to the main interface...",
                    metadata={"title": "🔄 Complete"},
                )
            )
            yield inner_history, main_history

        def like(data: gr.LikeData):
            print("User liked the response")
            print(f"Index: {data.index}, Liked: {data.liked}")

        # Create the Gradio interface
        with gr.Blocks() as demo:
            # Verification page (if enabled)
            verification_container = gr.Group(visible=require_verification)
            main_interface_container = gr.Group(visible=not require_verification)

            with verification_container:
                gr.Markdown("# Biomni A1 Agent - Access Verification")
                gr.Markdown("Please enter your access code to continue.")
                access_code_input = gr.Textbox(label="Access Code", type="password")
                access_error_msg = gr.Markdown(visible=False)
                verify_btn = gr.Button("Verify Access")
                verify_btn.click(
                    fn=verify_access_code,
                    inputs=[access_code_input],
                    outputs=[verification_container, main_interface_container, access_error_msg],
                )

            # Main interface
            with main_interface_container:
                with gr.Row():
                    with gr.Column(scale=1):
                        main_chatbot = gr.Chatbot(
                            label="Biomni A1 Agent",
                            type="messages",
                            height=800,
                            show_copy_button=True,
                            show_share_button=True,
                        )
                    with gr.Column(scale=1):
                        innerloop_chatbot = gr.Chatbot(
                            label="Biomni Executor",
                            type="messages",
                            height=800,
                            show_copy_button=True,
                            show_share_button=True,
                        )

                with gr.Row():
                    prompt_input = gr.MultimodalTextbox(
                        interactive=True,
                        file_count="multiple",
                        placeholder="Ask something or upload a file...",
                        show_label=False,
                    )

                # Bind submission
                prompt_input.submit(
                    generate_response,
                    [prompt_input, innerloop_chatbot, main_chatbot],
                    [innerloop_chatbot, main_chatbot],
                ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])
                main_chatbot.like(like)

        # Launch
        print(f"Launching Gradio demo on {server_name}:7860")
        demo.launch(share=share, server_name=server_name)
