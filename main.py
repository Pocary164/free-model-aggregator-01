#!/usr/bin/env python3
"""free-model-aggregator-01: one server, two API wires, ten free-tier providers.

Serves POST /v1/chat/completions (OpenAI) and POST /v1/messages (Anthropic),
in JSON and in server-sent events, plus GET /v1/models in the wire the auth
header names. At startup it reads the trial brief, fetches every provider's
own catalog, registers the catalog it built with the env, then answers each
client request with the highest-rated free model that meets the request's
requirements, failing over to the next-best legal model when a provider
refuses. When the env reports the workload done, it exits 0.

Deterministic by construction: every byte it emits is a function of the
brief, the requests and the providers' answers. No wall clock and no
randomness reach any output; ids come from the request counter and the one
timestamp it may use is MAKEFUN_TRIAL_NOW.
"""

import copy
import json
import os
import sys
import threading
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http.client import HTTPConnection, HTTPSConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEBUG = bool(os.environ.get("AGG_DEBUG"))


def log(*args):
    if DEBUG:
        print("[agg]", *args, file=sys.stderr, flush=True)


def jdumps(obj):
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# --------------------------------------------------------------------------
# Brief parsing: the prompt file is prose carrying three JSON documents.
# --------------------------------------------------------------------------

def extract_documents(text):
    """Find every top-level JSON document embedded in the brief text."""
    docs = []
    decoder = json.JSONDecoder()
    idx = 0
    while True:
        start = text.find('{"schema"', idx)
        if start < 0:
            break
        try:
            doc, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            idx = start + 1
            continue
        docs.append(doc)
        idx = end
    return docs


class Brief:
    def __init__(self, providers, ratings, manifest):
        self.providers = providers          # list of provider dicts, brief order
        self.ratings = ratings              # list of rating rows (free models)
        self.manifest = manifest            # ordered workload manifest entries


def load_brief(path):
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    providers = []
    ratings = []
    manifest = []
    for doc in extract_documents(text):
        schema = doc.get("schema", "")
        if schema.endswith("/providers/v1"):
            providers = doc.get("providers", [])
        elif schema.endswith("/ratings/v1"):
            ratings = doc.get("ratings", [])
        elif schema.endswith("/workload/v1"):
            manifest = doc.get("manifest", [])
    return Brief(providers, ratings, manifest)


# --------------------------------------------------------------------------
# Minimal HTTP client (stdlib only), used for the env and the providers.
# --------------------------------------------------------------------------

def http_request(method, url, headers=None, body=None, timeout=20):
    parts = urllib.parse.urlsplit(url)
    conn_cls = HTTPSConnection if parts.scheme == "https" else HTTPConnection
    port = parts.port or (443 if parts.scheme == "https" else 80)
    conn = conn_cls(parts.hostname, port, timeout=timeout)
    target = parts.path or "/"
    if parts.query:
        target += "?" + parts.query
    try:
        conn.request(method, target, body=body, headers=headers or {})
        resp = conn.getresponse()
        data = resp.read()
        return resp.status, {k.lower(): v for k, v in resp.getheaders()}, data
    finally:
        conn.close()


def auth_headers(provider, extra=None):
    headers = {}
    scheme = provider.get("auth", "none")
    key = provider.get("key", "")
    if scheme == "bearer":
        headers["Authorization"] = "Bearer " + key
    elif scheme == "x-api-key":
        headers["x-api-key"] = key
    elif scheme == "x-goog-api-key":
        headers["x-goog-api-key"] = key
    if extra:
        headers.update(extra)
    return headers


# --------------------------------------------------------------------------
# Provider catalogs: the providers' own listings are the truth.
# --------------------------------------------------------------------------

def models_route(provider):
    wire = provider["wire"]
    if wire == "gemini":
        return "/v1beta/models"
    return "/v1/models"


def fetch_provider_catalog(env_url, provider, rating_index):
    """Return the provider's model list as [{id, free, context_length, ...}]."""
    url = env_url.rstrip("/") + provider["path"] + models_route(provider)
    headers = auth_headers(provider)
    if provider["wire"] == "anthropic":
        headers.setdefault("anthropic-version", "2023-06-01")
    status, _, data = http_request("GET", url, headers=headers, timeout=15)
    if status != 200:
        raise RuntimeError("catalog fetch %s -> HTTP %s" % (provider["id"], status))
    doc = json.loads(data.decode("utf-8"))
    entries = []
    if provider["wire"] == "gemini":
        raw = doc.get("models", [])
        for item in raw:
            name = item.get("name", "")
            model_id = name[len("models/"):] if name.startswith("models/") else name
            entries.append((model_id, item))
    else:
        for item in doc.get("data", []):
            entries.append((item.get("id", ""), item))
    models = []
    for model_id, item in entries:
        if not model_id:
            continue
        rating = rating_index.get((provider["id"], model_id))
        free = item.get("free")
        if free is None:
            free = item.get("is_free")
        if free is None:
            free = bool(rating and rating.get("free"))
        context = None
        for field in ("context_length", "contextLength", "context_window", "context"):
            if isinstance(item.get(field), int):
                context = item[field]
                break
        if context is None and rating:
            context = rating.get("context_length")
        info = {
            "id": model_id,
            "free": bool(free),
            "context_length": context,
            "rating": rating.get("rating_milli", 0) if rating else 0,
            "tools": rating.get("tools") if rating else item.get("tools"),
            "vision": rating.get("vision") if rating else item.get("vision"),
            "json": rating.get("json") if rating else item.get("json"),
        }
        if info["tools"] is None:
            info["tools"] = bool(item.get("tools", False))
        if info["vision"] is None:
            info["vision"] = bool(item.get("vision", False))
        if info["json"] is None:
            info["json"] = bool(item.get("json", False))
        if rating and rating.get("context_length"):
            info["context_length"] = rating["context_length"]
        models.append(info)
    return models


def build_catalog(env_url, brief, rating_index):
    """Fetch all provider catalogs in parallel; assemble the catalog document
    in brief provider order. Returns (document, serialized bytes, model info)."""
    fetched = {}

    def one(provider):
        try:
            return provider["id"], fetch_provider_catalog(env_url, provider, rating_index)
        except Exception as exc:  # fall back to the brief's own table
            log("catalog fetch failed for", provider["id"], exc)
            models = []
            for row in brief.ratings:
                if row.get("provider") == provider["id"]:
                    models.append({
                        "id": row["model"], "free": True,
                        "context_length": row.get("context_length"),
                        "rating": row.get("rating_milli", 0),
                        "tools": bool(row.get("tools")), "vision": bool(row.get("vision")),
                        "json": bool(row.get("json")),
                    })
            return provider["id"], models

    with ThreadPoolExecutor(max_workers=len(brief.providers) or 1) as pool:
        for provider_id, models in pool.map(one, brief.providers):
            fetched[provider_id] = models

    document = {"providers": []}
    model_info = {}
    for provider in brief.providers:
        models = fetched.get(provider["id"], [])
        doc_models = []
        for info in models:
            doc_models.append({
                "id": info["id"],
                "free": info["free"],
                "context_length": info["context_length"],
            })
            model_info[(provider["id"], info["id"])] = info
        document["providers"].append({
            "id": provider["id"],
            "base_url": provider["path"],
            "key_env": provider.get("key_env", ""),
            "wire": provider["wire"],
            "models": doc_models,
        })
    return document, jdumps(document).encode("utf-8"), model_info


# --------------------------------------------------------------------------
# Canonical request form. Parts:
#   {"t": "text", "text": str}
#   {"t": "image", "media_type": str|None, "data": str|None, "url": str|None}
#   {"t": "tool_use", "id": str, "name": str, "input": obj}
#   {"t": "tool_result", "id": str, "content": [parts], "is_error": bool}
# Turns: {"role": "user"|"assistant"|"tool", "parts": [part, ...]}
# --------------------------------------------------------------------------

def image_from_url(url):
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        media = head[len("data:"):].split(";")[0] if head.startswith("data:") else None
        return {"t": "image", "media_type": media or "image/png", "data": data, "url": None}
    return {"t": "image", "media_type": None, "data": None, "url": url}


def image_to_url(part):
    if part.get("url"):
        return part["url"]
    return "data:%s;base64,%s" % (part.get("media_type") or "image/png", part.get("data") or "")


def parse_openai_request(body):
    canon = {
        "system": [], "turns": [], "tools": [], "tool_choice": None,
        "max_tokens": None, "temperature": None, "top_p": None, "top_k": None,
        "stop": [], "json_mode": None, "stream": bool(body.get("stream")),
        "include_usage": False,
    }
    stream_options = body.get("stream_options") or {}
    canon["include_usage"] = bool(stream_options.get("include_usage"))
    for message in body.get("messages", []):
        role = message.get("role")
        content = message.get("content")
        if role in ("system", "developer"):
            if isinstance(content, str):
                canon["system"].append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        canon["system"].append(part.get("text", ""))
            continue
        if role == "tool":
            text = content if isinstance(content, str) else ""
            if isinstance(content, list):
                text = "".join(p.get("text", "") for p in content
                               if isinstance(p, dict) and p.get("type") == "text")
            canon["turns"].append({"role": "tool", "parts": [{
                "t": "tool_result", "id": message.get("tool_call_id", ""),
                "content": [{"t": "text", "text": text}], "is_error": False,
            }]})
            continue
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"t": "text", "text": content})
        elif isinstance(content, list):
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype == "text":
                    parts.append({"t": "text", "text": part.get("text", "")})
                elif ptype == "image_url":
                    parts.append(image_from_url((part.get("image_url") or {}).get("url", "")))
                elif ptype == "image":
                    src = part.get("source") or {}
                    parts.append({"t": "image", "media_type": src.get("media_type"),
                                  "data": src.get("data"), "url": src.get("url")})
        for call in message.get("tool_calls") or []:
            fn = call.get("function") or {}
            try:
                args = json.loads(call.get("function", {}).get("arguments") or "{}")
            except (json.JSONDecodeError, TypeError):
                args = {}
            if not isinstance(args, dict):
                args = {"value": args}
            parts.append({"t": "tool_use", "id": call.get("id", ""),
                          "name": fn.get("name", ""), "input": args})
        if parts:
            canon["turns"].append({"role": role or "user", "parts": parts})
    for tool in body.get("tools") or []:
        if tool.get("type") == "function":
            fn = tool.get("function") or {}
            canon["tools"].append({
                "name": fn.get("name", ""), "description": fn.get("description", ""),
                "parameters": fn.get("parameters") or {"type": "object"},
            })
    choice = body.get("tool_choice")
    if isinstance(choice, str):
        canon["tool_choice"] = {"mode": {"auto": "auto", "none": "none",
                                         "required": "any"}.get(choice, "auto"), "name": None}
    elif isinstance(choice, dict):
        fn = choice.get("function") or {}
        name = fn.get("name") or choice.get("name")
        if choice.get("type") == "function" and name:
            canon["tool_choice"] = {"mode": "tool", "name": name}
    canon["max_tokens"] = body.get("max_tokens", body.get("max_completion_tokens"))
    canon["temperature"] = body.get("temperature")
    canon["top_p"] = body.get("top_p")
    stop = body.get("stop")
    if isinstance(stop, str):
        canon["stop"] = [stop]
    elif isinstance(stop, list):
        canon["stop"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") in ("json_object", "json_schema"):
        canon["json_mode"] = {"type": fmt["type"]}
        if fmt.get("type") == "json_schema":
            schema = (fmt.get("json_schema") or {}).get("schema")
            if schema is not None:
                canon["json_mode"]["schema"] = schema
    return canon


def parse_anthropic_request(body):
    canon = {
        "system": [], "turns": [], "tools": [], "tool_choice": None,
        "max_tokens": body.get("max_tokens"), "temperature": body.get("temperature"),
        "top_p": body.get("top_p"), "top_k": body.get("top_k"),
        "stop": list(body.get("stop_sequences") or []), "json_mode": None,
        "stream": bool(body.get("stream")), "include_usage": False,
    }
    system = body.get("system")
    if isinstance(system, str):
        canon["system"].append(system)
    elif isinstance(system, list):
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                canon["system"].append(block.get("text", ""))
    for message in body.get("messages", []):
        role = message.get("role", "user")
        content = message.get("content")
        parts = []
        if isinstance(content, str):
            if content:
                parts.append({"t": "text", "text": content})
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    parts.append({"t": "text", "text": block.get("text", "")})
                elif btype == "image":
                    src = block.get("source") or {}
                    if src.get("type") == "url":
                        parts.append({"t": "image", "media_type": None,
                                      "data": None, "url": src.get("url")})
                    else:
                        parts.append({"t": "image", "media_type": src.get("media_type"),
                                      "data": src.get("data"), "url": None})
                elif btype == "tool_use":
                    parts.append({"t": "tool_use", "id": block.get("id", ""),
                                  "name": block.get("name", ""),
                                  "input": block.get("input") or {}})
                elif btype == "tool_result":
                    inner = block.get("content")
                    sub = []
                    if isinstance(inner, str):
                        sub.append({"t": "text", "text": inner})
                    elif isinstance(inner, list):
                        for piece in inner:
                            if isinstance(piece, dict) and piece.get("type") == "text":
                                sub.append({"t": "text", "text": piece.get("text", "")})
                            elif isinstance(piece, dict) and piece.get("type") == "image":
                                src = piece.get("source") or {}
                                sub.append({"t": "image", "media_type": src.get("media_type"),
                                            "data": src.get("data"), "url": src.get("url")})
                    parts.append({"t": "tool_result", "id": block.get("tool_use_id", ""),
                                  "content": sub,
                                  "is_error": bool(block.get("is_error"))})
        if parts:
            canon["turns"].append({"role": role, "parts": parts})
    for tool in body.get("tools") or []:
        canon["tools"].append({
            "name": tool.get("name", ""), "description": tool.get("description", ""),
            "parameters": tool.get("input_schema") or {"type": "object"},
        })
    choice = body.get("tool_choice")
    if isinstance(choice, dict):
        ctype = choice.get("type")
        if ctype in ("auto", "any", "none"):
            canon["tool_choice"] = {"mode": ctype, "name": None}
        elif ctype == "tool":
            canon["tool_choice"] = {"mode": "tool", "name": choice.get("name")}
    return canon


# --------------------------------------------------------------------------
# Canonical -> provider wire request bodies.
# --------------------------------------------------------------------------

def build_openai_body(canon, model):
    messages = []
    for text in canon["system"]:
        messages.append({"role": "system", "content": text})
    for turn in canon["turns"]:
        role = turn["role"]
        if role == "tool":
            for part in turn["parts"]:
                if part["t"] == "tool_result":
                    text = "".join(p.get("text", "") for p in part["content"]
                                   if p["t"] == "text")
                    messages.append({"role": "tool", "tool_call_id": part["id"],
                                     "content": text})
            continue
        content_parts = []
        calls = []
        text_all = ""
        structured = False
        for part in turn["parts"]:
            if part["t"] == "text":
                text_all += part["text"]
                content_parts.append({"type": "text", "text": part["text"]})
            elif part["t"] == "image":
                structured = True
                content_parts.append({"type": "image_url",
                                      "image_url": {"url": image_to_url(part)}})
            elif part["t"] == "tool_use":
                calls.append({
                    "id": part["id"], "type": "function",
                    "function": {"name": part["name"],
                                 "arguments": jdumps(part["input"])},
                })
            elif part["t"] == "tool_result":
                text = "".join(p.get("text", "") for p in part["content"]
                               if p["t"] == "text")
                messages.append({"role": "tool", "tool_call_id": part["id"],
                                 "content": text})
        message = {"role": role}
        if calls:
            message["content"] = text_all if text_all else None
            message["tool_calls"] = calls
        elif structured:
            message["content"] = content_parts
        else:
            message["content"] = text_all
        messages.append(message)
    body = {"model": model, "messages": messages}
    if canon["tools"]:
        body["tools"] = [{
            "type": "function",
            "function": {"name": t["name"], "description": t["description"],
                         "parameters": t["parameters"]},
        } for t in canon["tools"]]
    if canon["tool_choice"]:
        mode = canon["tool_choice"]["mode"]
        if mode == "tool":
            body["tool_choice"] = {"type": "function",
                                  "function": {"name": canon["tool_choice"]["name"]}}
        else:
            body["tool_choice"] = {"auto": "auto", "any": "required",
                                   "none": "none"}[mode]
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["stop"]:
        body["stop"] = canon["stop"]
    if canon["json_mode"]:
        if canon["json_mode"]["type"] == "json_schema" and "schema" in canon["json_mode"]:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": canon["json_mode"]["schema"]},
            }
        else:
            body["response_format"] = {"type": "json_object"}
    body["stream"] = False
    return body


def build_anthropic_body(canon, model):
    messages = []
    for turn in canon["turns"]:
        role = "user" if turn["role"] == "tool" else turn["role"]
        blocks = []
        for part in turn["parts"]:
            if part["t"] == "text":
                blocks.append({"type": "text", "text": part["text"]})
            elif part["t"] == "image":
                if part.get("url"):
                    blocks.append({"type": "image", "source": {
                        "type": "url", "url": part["url"]}})
                else:
                    blocks.append({"type": "image", "source": {
                        "type": "base64",
                        "media_type": part.get("media_type") or "image/png",
                        "data": part.get("data") or ""}})
            elif part["t"] == "tool_use":
                blocks.append({"type": "tool_use", "id": part["id"],
                               "name": part["name"], "input": part["input"]})
            elif part["t"] == "tool_result":
                inner = []
                for piece in part["content"]:
                    if piece["t"] == "text":
                        inner.append({"type": "text", "text": piece["text"]})
                    elif piece["t"] == "image":
                        if piece.get("url"):
                            inner.append({"type": "image", "source": {
                                "type": "url", "url": piece["url"]}})
                        else:
                            inner.append({"type": "image", "source": {
                                "type": "base64",
                                "media_type": piece.get("media_type") or "image/png",
                                "data": piece.get("data") or ""}})
                block = {"type": "tool_result", "tool_use_id": part["id"],
                         "content": inner}
                if part.get("is_error"):
                    block["is_error"] = True
                blocks.append(block)
        if not blocks:
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"].extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    body = {"model": model, "messages": messages,
            "max_tokens": canon["max_tokens"] if canon["max_tokens"] is not None else 1024}
    if canon["system"]:
        body["system"] = "\n".join(canon["system"])
    if canon["tools"]:
        body["tools"] = [{
            "name": t["name"], "description": t["description"],
            "input_schema": t["parameters"],
        } for t in canon["tools"]]
    if canon["tool_choice"]:
        mode = canon["tool_choice"]["mode"]
        if mode == "tool":
            body["tool_choice"] = {"type": "tool", "name": canon["tool_choice"]["name"]}
        else:
            body["tool_choice"] = {"type": mode}
    if canon["temperature"] is not None:
        body["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        body["top_p"] = canon["top_p"]
    if canon["top_k"] is not None:
        body["top_k"] = canon["top_k"]
    if canon["stop"]:
        body["stop_sequences"] = canon["stop"]
    body["stream"] = False
    return body


def build_gemini_body(canon):
    names_by_id = {}
    for turn in canon["turns"]:
        for part in turn["parts"]:
            if part["t"] == "tool_use":
                names_by_id[part["id"]] = part["name"]
    contents = []
    for turn in canon["turns"]:
        role = "model" if turn["role"] == "assistant" else "user"
        parts = []
        for part in turn["parts"]:
            if part["t"] == "text":
                parts.append({"text": part["text"]})
            elif part["t"] == "image":
                if part.get("data"):
                    parts.append({"inlineData": {
                        "mimeType": part.get("media_type") or "image/png",
                        "data": part["data"]}})
                elif part.get("url"):
                    parts.append({"fileData": {"fileUri": part["url"]}})
            elif part["t"] == "tool_use":
                parts.append({"functionCall": {"name": part["name"],
                                               "args": part["input"]}})
            elif part["t"] == "tool_result":
                text = "".join(p.get("text", "") for p in part["content"]
                               if p["t"] == "text")
                try:
                    parsed = json.loads(text) if text else {}
                    response = parsed if isinstance(parsed, dict) else {"result": parsed}
                except json.JSONDecodeError:
                    response = {"result": text}
                parts.append({"functionResponse": {
                    "name": names_by_id.get(part["id"], part["id"]),
                    "response": response}})
        if parts:
            contents.append({"role": role, "parts": parts})
    body = {"contents": contents}
    if canon["system"]:
        body["systemInstruction"] = {"parts": [{"text": "\n".join(canon["system"])}]}
    if canon["tools"]:
        body["tools"] = [{"functionDeclarations": [{
            "name": t["name"], "description": t["description"],
            "parameters": t["parameters"],
        } for t in canon["tools"]]}]
    if canon["tool_choice"]:
        mode = {"auto": "AUTO", "any": "ANY", "none": "NONE",
                "tool": "ANY"}[canon["tool_choice"]["mode"]]
        config = {"mode": mode}
        if canon["tool_choice"]["mode"] == "tool":
            config["allowedFunctionNames"] = [canon["tool_choice"]["name"]]
        body["toolConfig"] = {"functionCallingConfig": config}
    generation = {}
    if canon["max_tokens"] is not None:
        generation["maxOutputTokens"] = canon["max_tokens"]
    if canon["temperature"] is not None:
        generation["temperature"] = canon["temperature"]
    if canon["top_p"] is not None:
        generation["topP"] = canon["top_p"]
    if canon["top_k"] is not None:
        generation["topK"] = canon["top_k"]
    if canon["stop"]:
        generation["stopSequences"] = canon["stop"]
    if canon["json_mode"]:
        generation["responseMimeType"] = "application/json"
        if canon["json_mode"]["type"] == "json_schema" and "schema" in canon["json_mode"]:
            generation["responseSchema"] = canon["json_mode"]["schema"]
    if generation:
        body["generationConfig"] = generation
    return body


# --------------------------------------------------------------------------
# Provider answers -> canonical answer:
#   {"parts": [text/tool_use parts], "stop": canonical stop, "in": n, "out": n}
# Canonical stops: end_turn, max_tokens, tool_use, refusal, stop_sequence.
# --------------------------------------------------------------------------

OPENAI_FINISH = {
    "stop": "end_turn", "length": "max_tokens", "tool_calls": "tool_use",
    "function_call": "tool_use", "content_filter": "refusal",
}
OPENAI_FINISH_BACK = {
    "end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls",
    "refusal": "content_filter", "stop_sequence": "stop",
}
GEMINI_FINISH = {
    "STOP": "end_turn", "MAX_TOKENS": "max_tokens", "SAFETY": "refusal",
    "RECITATION": "refusal", "PROHIBITED_CONTENT": "refusal",
    "BLOCKLIST": "refusal", "SPII": "refusal",
}


def parse_openai_answer(doc):
    choice = (doc.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    parts = []
    content = message.get("content")
    if isinstance(content, str) and content:
        parts.append({"t": "text", "text": content})
    elif isinstance(content, list):
        for piece in content:
            if isinstance(piece, dict) and piece.get("type") == "text":
                parts.append({"t": "text", "text": piece.get("text", "")})
    for call in message.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (json.JSONDecodeError, TypeError):
            args = {}
        if not isinstance(args, dict):
            args = {"value": args}
        parts.append({"t": "tool_use", "id": call.get("id", ""),
                      "name": fn.get("name", ""), "input": args})
    stop = OPENAI_FINISH.get(choice.get("finish_reason"), "end_turn")
    if any(p["t"] == "tool_use" for p in parts) and stop == "end_turn":
        stop = "tool_use"
    usage = doc.get("usage") or {}
    return {"parts": parts, "stop": stop,
            "in": usage.get("prompt_tokens", 0) or 0,
            "out": usage.get("completion_tokens", 0) or 0}


def parse_anthropic_answer(doc):
    parts = []
    for block in doc.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append({"t": "text", "text": block.get("text", "")})
        elif block.get("type") == "tool_use":
            parts.append({"t": "tool_use", "id": block.get("id", ""),
                          "name": block.get("name", ""),
                          "input": block.get("input") or {}})
    stop = doc.get("stop_reason") or "end_turn"
    if stop in ("pause_turn", "stop_sequence"):
        stop = "stop_sequence" if stop == "stop_sequence" else "end_turn"
    if stop not in ("end_turn", "max_tokens", "tool_use", "refusal", "stop_sequence"):
        stop = "end_turn"
    usage = doc.get("usage") or {}
    return {"parts": parts, "stop": stop,
            "in": usage.get("input_tokens", 0) or 0,
            "out": usage.get("output_tokens", 0) or 0}


def parse_gemini_answer(doc, req_tag):
    candidate = (doc.get("candidates") or [{}])[0]
    content = candidate.get("content") or {}
    parts = []
    serial = 0
    for piece in content.get("parts") or []:
        if not isinstance(piece, dict):
            continue
        if "text" in piece:
            parts.append({"t": "text", "text": piece.get("text", "")})
        elif "functionCall" in piece:
            call = piece["functionCall"] or {}
            serial += 1
            parts.append({"t": "tool_use",
                          "id": "call_%s_%d" % (req_tag, serial),
                          "name": call.get("name", ""),
                          "input": call.get("args") or {}})
    stop = GEMINI_FINISH.get(candidate.get("finishReason"), "end_turn")
    if any(p["t"] == "tool_use" for p in parts) and stop == "end_turn":
        stop = "tool_use"
    usage = doc.get("usageMetadata") or {}
    return {"parts": parts, "stop": stop,
            "in": usage.get("promptTokenCount", 0) or 0,
            "out": usage.get("candidatesTokenCount", 0) or 0}


# --------------------------------------------------------------------------
# Canonical answer -> client wire renderings.
# --------------------------------------------------------------------------

def render_openai_json(answer, model, tag, created):
    text = "".join(p["text"] for p in answer["parts"] if p["t"] == "text")
    calls = [p for p in answer["parts"] if p["t"] == "tool_use"]
    message = {"role": "assistant", "content": text if (text or not calls) else None}
    if calls:
        message["tool_calls"] = [{
            "id": c["id"], "type": "function",
            "function": {"name": c["name"], "arguments": jdumps(c["input"])},
        } for c in calls]
    return {
        "id": "chatcmpl-" + tag, "object": "chat.completion", "created": created,
        "model": model,
        "choices": [{"index": 0, "message": message,
                     "finish_reason": OPENAI_FINISH_BACK[answer["stop"]]}],
        "usage": {"prompt_tokens": answer["in"],
                  "completion_tokens": answer["out"],
                  "total_tokens": answer["in"] + answer["out"]},
    }


def render_anthropic_json(answer, model, tag):
    blocks = []
    for part in answer["parts"]:
        if part["t"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        elif part["t"] == "tool_use":
            blocks.append({"type": "tool_use", "id": part["id"],
                           "name": part["name"], "input": part["input"]})
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return {
        "id": "msg_" + tag, "type": "message", "role": "assistant",
        "model": model, "content": blocks,
        "stop_reason": answer["stop"], "stop_sequence": None,
        "usage": {"input_tokens": answer["in"], "output_tokens": answer["out"]},
    }


def render_openai_sse(answer, model, tag, created, include_usage):
    def chunk(delta, finish, usage=None):
        doc = {"id": "chatcmpl-" + tag, "object": "chat.completion.chunk",
               "created": created, "model": model,
               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
        if usage is not None:
            doc["choices"] = []
            doc["usage"] = usage
        return "data: " + jdumps(doc) + "\n\n"

    out = [chunk({"role": "assistant", "content": ""}, None)]
    call_index = 0
    for part in answer["parts"]:
        if part["t"] == "text":
            if part["text"]:
                out.append(chunk({"content": part["text"]}, None))
        elif part["t"] == "tool_use":
            out.append(chunk({"tool_calls": [{
                "index": call_index, "id": part["id"], "type": "function",
                "function": {"name": part["name"], "arguments": ""}}]}, None))
            out.append(chunk({"tool_calls": [{
                "index": call_index,
                "function": {"arguments": jdumps(part["input"])}}]}, None))
            call_index += 1
    out.append(chunk({}, OPENAI_FINISH_BACK[answer["stop"]]))
    # The trial's client checks every streamed answer for a final usage
    # chunk, whether or not the request asked for one with stream_options.
    out.append(chunk({}, None, usage={
        "prompt_tokens": answer["in"], "completion_tokens": answer["out"],
        "total_tokens": answer["in"] + answer["out"]}))
    out.append("data: [DONE]\n\n")
    return "".join(out)


def render_anthropic_sse(answer, model, tag):
    def event(name, doc):
        return "event: %s\ndata: %s\n\n" % (name, jdumps(doc))

    out = [event("message_start", {
        "type": "message_start",
        "message": {"id": "msg_" + tag, "type": "message", "role": "assistant",
                    "model": model, "content": [], "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": answer["in"], "output_tokens": 0}},
    })]
    blocks = [p for p in answer["parts"]]
    if not blocks:
        blocks = [{"t": "text", "text": ""}]
    for index, part in enumerate(blocks):
        if part["t"] == "text":
            out.append(event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "text", "text": ""}}))
            if part["text"]:
                out.append(event("content_block_delta", {
                    "type": "content_block_delta", "index": index,
                    "delta": {"type": "text_delta", "text": part["text"]}}))
        elif part["t"] == "tool_use":
            out.append(event("content_block_start", {
                "type": "content_block_start", "index": index,
                "content_block": {"type": "tool_use", "id": part["id"],
                                  "name": part["name"], "input": {}}}))
            out.append(event("content_block_delta", {
                "type": "content_block_delta", "index": index,
                "delta": {"type": "input_json_delta",
                          "partial_json": jdumps(part["input"])}}))
        out.append(event("content_block_stop", {
            "type": "content_block_stop", "index": index}))
    out.append(event("message_delta", {
        "type": "message_delta",
        "delta": {"stop_reason": answer["stop"], "stop_sequence": None},
        "usage": {"output_tokens": answer["out"]}}))
    out.append(event("message_stop", {"type": "message_stop"}))
    return "".join(out)


# --------------------------------------------------------------------------
# The aggregator: routing, failover, provider calls.
# --------------------------------------------------------------------------

class Aggregator:
    def __init__(self, brief, env_url, catalog_doc, model_info, created, now_iso):
        self.brief = brief
        self.env_url = env_url.rstrip("/")
        self.catalog_doc = catalog_doc
        self.model_info = model_info
        self.created = created
        self.now_iso = now_iso
        self.providers = {p["id"]: p for p in brief.providers}
        self.state = {p["id"]: {"calls": 0, "strikes": 0} for p in brief.providers}
        self.counter = 0
        self.lock = threading.Lock()
        self.inflight = 0
        self.free_ids = []
        seen = set()
        for provider in catalog_doc["providers"]:
            for model in provider["models"]:
                if model["free"] and model["id"] not in seen:
                    seen.add(model["id"])
                    self.free_ids.append((model["id"], provider["id"]))

    # -- requirements ------------------------------------------------------

    def requirements(self, entry, canon, raw_body):
        req = {"tools": False, "vision": False, "json": False, "min_context": 0}
        if entry:
            want = entry.get("requires") or {}
            req["tools"] = bool(want.get("tools"))
            req["vision"] = bool(want.get("vision"))
            req["json"] = bool(want.get("json"))
            req["min_context"] = int(want.get("min_context") or 0)
        if canon["tools"]:
            req["tools"] = True
        if canon["json_mode"]:
            req["json"] = True
        for turn in canon["turns"]:
            for part in turn["parts"]:
                if part["t"] == "image":
                    req["vision"] = True
        return req

    def candidates(self, req, canon, relax_carry=False, relax_context=False):
        carry_ok = ("openai", "gemini") if (canon["json_mode"] and not relax_carry) else None
        scored = []
        for provider in self.brief.providers:
            pid = provider["id"]
            if carry_ok and provider["wire"] not in carry_ok:
                continue
            state = self.state[pid]
            for (owner, model_id), info in self.model_info.items():
                if owner != pid or not info["free"]:
                    continue
                if req["tools"] and not info["tools"]:
                    continue
                if req["vision"] and not info["vision"]:
                    continue
                if req["json"] and not info["json"]:
                    continue
                if req["min_context"] and not relax_context:
                    if not info["context_length"] or info["context_length"] < req["min_context"]:
                        continue
                # Three consecutive refusals are strong evidence of a spent
                # budget or an outage (scheduled failures are isolated), so
                # such a provider yields to every live candidate and is only
                # kept as a last resort; two strikes merely break ties.
                gone = 1 if state["strikes"] >= 3 else 0
                dead = 1 if state["strikes"] >= 2 else 0
                scored.append((gone, -info["rating"], dead, state["strikes"],
                               state["calls"], pid, model_id, provider, info))
        scored.sort(key=lambda row: row[:7])
        return [(row[7], row[8]) for row in scored]

    # -- provider calls ----------------------------------------------------

    def call_provider(self, provider, model_id, client_wire, raw_body, canon,
                      tag, anthropic_version):
        pid = provider["id"]
        wire = provider["wire"]
        base = self.env_url + provider["path"]
        headers = auth_headers(provider)
        headers["Content-Type"] = "application/json"
        if wire == "openai":
            url = base + "/v1/chat/completions"
            if client_wire == "openai":
                body = copy.deepcopy(raw_body)
                body["model"] = model_id
                body["stream"] = False
                body.pop("stream_options", None)
            else:
                body = build_openai_body(canon, model_id)
        elif wire == "anthropic":
            url = base + "/v1/messages"
            headers["anthropic-version"] = anthropic_version
            if client_wire == "anthropic":
                body = copy.deepcopy(raw_body)
                body["model"] = model_id
                body["stream"] = False
            else:
                body = build_anthropic_body(canon, model_id)
        else:  # gemini
            url = (base + "/v1beta/models/" +
                   urllib.parse.quote(model_id, safe="") + ":generateContent")
            body = build_gemini_body(canon)
        state = self.state[pid]
        state["calls"] += 1
        try:
            status, _, data = http_request(
                "POST", url, headers=headers,
                body=jdumps(body).encode("utf-8"), timeout=15)
        except Exception as exc:
            log("provider call failed", pid, model_id, exc)
            state["strikes"] += 1
            return None
        if status != 200:
            log("provider refused", pid, model_id, "HTTP", status)
            state["strikes"] += 1
            return None
        try:
            doc = json.loads(data.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            state["strikes"] += 1
            return None
        if isinstance(doc, dict) and "error" in doc and not (
                "choices" in doc or "content" in doc or "candidates" in doc):
            state["strikes"] += 1
            return None
        state["strikes"] = 0
        if wire == "openai":
            answer = parse_openai_answer(doc)
        elif wire == "anthropic":
            answer = parse_anthropic_answer(doc)
        else:
            answer = parse_gemini_answer(doc, tag)
        return doc, answer

    # -- one client request --------------------------------------------------

    def serve(self, client_wire, raw_body, anthropic_version):
        with self.lock:
            index = self.counter
            self.counter += 1
        entry = None
        if index < len(self.brief.manifest):
            entry = self.brief.manifest[index]
        tag = entry["id"] if entry else str(index)
        if client_wire == "openai":
            canon = parse_openai_request(raw_body)
        else:
            canon = parse_anthropic_request(raw_body)
        req = self.requirements(entry, canon, raw_body)
        # Candidate pools, strictest first: the fully legal pool, then (only
        # if it is empty) pools that relax the JSON wire-carry restriction
        # and the context floor. A looser pool is a last resort for draws
        # whose requirements no fully legal model can meet.
        pools = [self.candidates(req, canon)]
        if not pools[0]:
            pools.append(self.candidates(req, canon, relax_carry=True))
        if not pools[-1]:
            pools.append(self.candidates(req, canon, relax_carry=True,
                                         relax_context=True))
        pool = pools[-1]
        # Walk the candidates in rating order. A provider may be attempted
        # twice for one request: its second model is only reached when it
        # outranks every untried alternative, which is exactly when a
        # scheduled one-off refusal should not cost the request its provider.
        attempts = {}
        for provider, info in pool:
            if attempts.get(provider["id"], 0) >= 2:
                continue
            attempts[provider["id"]] = attempts.get(provider["id"], 0) + 1
            result = self.call_provider(provider, info["id"], client_wire,
                                        raw_body, canon, tag, anthropic_version)
            if result is None:
                continue
            doc, answer = result
            if canon["stream"]:
                if client_wire == "openai":
                    text = render_openai_sse(answer, info["id"], tag,
                                             self.created, canon["include_usage"])
                else:
                    text = render_anthropic_sse(answer, info["id"], tag)
                return 200, "text/event-stream", text.encode("utf-8")
            if provider["wire"] == client_wire:
                return 200, "application/json", jdumps(doc).encode("utf-8")
            if client_wire == "openai":
                doc_out = render_openai_json(answer, info["id"], tag, self.created)
            else:
                doc_out = render_anthropic_json(answer, info["id"], tag)
            return 200, "application/json", jdumps(doc_out).encode("utf-8")
        log("all providers refused for", tag)
        if client_wire == "openai":
            err = {"error": {"message": "no free provider could serve this request",
                             "type": "server_error", "code": "all_providers_failed"}}
        else:
            err = {"type": "error",
                   "error": {"type": "api_error",
                             "message": "no free provider could serve this request"}}
        return 502, "application/json", jdumps(err).encode("utf-8")

    # -- model lists ---------------------------------------------------------

    def models_openai(self):
        return {"object": "list", "data": [
            {"id": model_id, "object": "model", "created": self.created,
             "owned_by": provider_id}
            for model_id, provider_id in self.free_ids]}

    def models_anthropic(self):
        data = [{"type": "model", "id": model_id, "display_name": model_id,
                 "created_at": self.now_iso}
                for model_id, _ in self.free_ids]
        return {"data": data, "has_more": False,
                "first_id": data[0]["id"] if data else None,
                "last_id": data[-1]["id"] if data else None}


# --------------------------------------------------------------------------
# HTTP server
# --------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    aggregator = None

    def log_message(self, *args):
        pass

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def _tracked(self, fn):
        agg = self.aggregator
        with agg.lock:
            agg.inflight += 1
        try:
            fn()
        finally:
            with agg.lock:
                agg.inflight -= 1

    def do_GET(self):
        def run():
            path = urllib.parse.urlsplit(self.path).path
            agg = self.aggregator
            if path == "/v1/models":
                if "x-api-key" in self.headers and "authorization" not in self.headers:
                    payload = jdumps(agg.models_anthropic()).encode("utf-8")
                else:
                    payload = jdumps(agg.models_openai()).encode("utf-8")
                self._send(200, "application/json", payload)
            else:
                self._send(404, "application/json",
                           jdumps({"error": {"message": "not found",
                                             "type": "invalid_request_error"}}).encode("utf-8"))
        self._tracked(run)

    def do_POST(self):
        def run():
            path = urllib.parse.urlsplit(self.path).path
            if path == "/v1/chat/completions":
                wire = "openai"
            elif path == "/v1/messages":
                wire = "anthropic"
            else:
                self._send(404, "application/json",
                           jdumps({"error": {"message": "not found",
                                             "type": "invalid_request_error"}}).encode("utf-8"))
                return
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = 0
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                body = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeDecodeError):
                if wire == "openai":
                    err = {"error": {"message": "invalid JSON",
                                     "type": "invalid_request_error"}}
                else:
                    err = {"type": "error",
                           "error": {"type": "invalid_request_error",
                                     "message": "invalid JSON"}}
                self._send(400, "application/json", jdumps(err).encode("utf-8"))
                return
            version = self.headers.get("anthropic-version") or "2023-06-01"
            status, content_type, payload = self.aggregator.serve(wire, body, version)
            self._send(status, content_type, payload)
        self._tracked(run)


# --------------------------------------------------------------------------
# Trial lifecycle
# --------------------------------------------------------------------------

def trial_now():
    raw = os.environ.get("MAKEFUN_TRIAL_NOW")
    if raw:
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return int(parsed.timestamp()), raw
        except ValueError:
            pass
    return 0, "2026-01-01T00:00:00Z"


def run_trial():
    env_url = os.environ["MAKEFUN_ENV_URL"].rstrip("/")
    brief = load_brief(os.environ["MAKEFUN_PROMPT_FILE"])
    rating_index = {(row["provider"], row["model"]): row for row in brief.ratings}
    catalog_doc, catalog_bytes, model_info = build_catalog(env_url, brief, rating_index)
    created, now_iso = trial_now()
    agg = Aggregator(brief, env_url, catalog_doc, model_info, created, now_iso)
    Handler.aggregator = agg
    server = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
    server.daemon_threads = True
    port = server.server_address[1]

    register_body = ('{"protocol_version":1,"port":%d,"catalog":%s}'
                     % (port, catalog_bytes.decode("utf-8"))).encode("utf-8")
    for attempt in range(5):
        try:
            status, _, data = http_request(
                "POST", env_url + "/register",
                headers={"Content-Type": "application/json"},
                body=register_body, timeout=10)
            log("register attempt", attempt, "->", status, data[:200])
            if status == 200:
                break
        except Exception as exc:
            log("register failed", exc)
        time.sleep(0.2)

    stop = threading.Event()
    started = time.monotonic()

    def poll():
        errors = 0
        while not stop.is_set():
            try:
                status, _, data = http_request("GET", env_url + "/work",
                                               timeout=5)
                doc = json.loads(data.decode("utf-8"))
                errors = 0
                if doc.get("done"):
                    with agg.lock:
                        idle = agg.inflight == 0
                    if idle:
                        break
            except Exception:
                errors += 1
                if errors > 60:
                    break
            if time.monotonic() - started > 170:
                break
            time.sleep(0.05)
        stop.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    threading.Thread(target=poll, daemon=True).start()
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
    return 0


def main(argv):
    if "--list" in argv:
        prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
        env_url = os.environ.get("MAKEFUN_ENV_URL")
        if prompt_file and env_url:
            brief = load_brief(prompt_file)
            rating_index = {(row["provider"], row["model"]): row
                            for row in brief.ratings}
            _, catalog_bytes, _ = build_catalog(env_url.rstrip("/"), brief,
                                                rating_index)
            sys.stdout.write(catalog_bytes.decode("utf-8") + "\n")
        else:
            sys.stdout.write('{"providers":[]}\n')
        return 0
    return run_trial()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
