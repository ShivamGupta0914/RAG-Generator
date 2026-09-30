from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app import graph as graph_module
from app.graph import (
    _lock,
    ask,
    create_chat,
    delete_chat,
    list_chats,
    read_thread,
    rename_chat,
    select_chat,
    snapshot,
)
from app.store import index_uploads, list_files

ROOT = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(ROOT / "templates"))

app = FastAPI(title="RAG Generator")


class Question(BaseModel):
    question: str
    chat_id: str | None = None


class Title(BaseModel):
    title: str


def _missing(exc: KeyError):
    raise HTTPException(status_code=404, detail=str(exc)) from exc


def _model_error(exc: Exception) -> str:
    text = str(exc)
    lowered = text.lower()
    if "overloaded" in lowered or "503" in lowered:
        return "The chat model is busy right now. Try again in a moment."
    if "429" in lowered or "rate limit" in lowered:
        return "The chat model is rate limited. Wait a moment and try again."
    return text


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.get("/api/chats")
def get_chats():
    return snapshot()


@app.post("/api/chats")
def post_chat():
    with _lock:
        create_chat()
        return {
            "active_chat_id": graph_module.active_chat_id,
            "chats": list_chats(),
            "files": list_files(),
            "messages": read_thread(),
        }


@app.post("/api/chats/{chat_id}/select")
def post_select(chat_id: str):
    with _lock:
        try:
            select_chat(chat_id)
        except KeyError as exc:
            _missing(exc)
        return {
            "active_chat_id": graph_module.active_chat_id,
            "chats": list_chats(),
            "files": list_files(),
            "messages": read_thread(),
        }


@app.post("/api/chats/{chat_id}/title")
def post_title(chat_id: str, body: Title):
    with _lock:
        try:
            rename_chat(chat_id, body.title)
        except KeyError as exc:
            _missing(exc)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {
            "active_chat_id": graph_module.active_chat_id,
            "chats": list_chats(),
            "files": list_files(),
            "messages": read_thread(),
        }


@app.delete("/api/chats/{chat_id}")
def remove_chat(chat_id: str):
    with _lock:
        try:
            delete_chat(chat_id)
        except KeyError as exc:
            _missing(exc)
        return {
            "active_chat_id": graph_module.active_chat_id,
            "chats": list_chats(),
            "files": list_files(),
            "messages": read_thread(),
        }


def _index_and_reset(chat_id: str | None, payload: list[tuple[str, bytes]]):
    with _lock:
        if chat_id:
            try:
                select_chat(chat_id)
            except KeyError as exc:
                _missing(exc)
        try:
            saved = index_uploads(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not read that file. {exc}") from exc

        return {
            "files": saved["files"],
            "chunks": saved["chunks"],
            "all_files": list_files(),
            "messages": read_thread(),
            "active_chat_id": graph_module.active_chat_id,
            "chats": list_chats(),
        }


@app.post("/api/upload")
async def upload(
    files: list[UploadFile] = File(...),
    chat_id: str | None = Form(None),
):
    payload = []
    for file in files:
        payload.append((file.filename or "file.txt", await file.read()))
    return await run_in_threadpool(_index_and_reset, chat_id, payload)


@app.post("/api/ask")
def ask_question(body: Question):
    try:
        messages = ask(body.question, body.chat_id)
    except KeyError as exc:
        _missing(exc)
    except ValueError as exc:
        if str(exc) == "Enter a question.":
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=_model_error(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=_model_error(exc)) from exc
    return {"messages": messages}
