# RAG Generator — development plan

A local document chat app. You upload `.txt`, `.md`, or `.pdf` files in the browser and ask questions in the same conversation. Answers come from those files. If the files do not contain the answer, the agent says it does not know.

## Problem statement

The app has to:

1. Accept documents at runtime.
2. Create a RAG application over those documents.
3. Let the user ask questions and receive grounded answers.
4. Work with different document sets without code changes.

Point 4 is the constraint that shapes the design. No file path, corpus, or document title is hardcoded. The same process indexes whatever `.txt`, `.md`, or `.pdf` files arrive at upload time, stores them in one Chroma collection, and answers from that collection. A leave policy, an invoice, or a handbook all use the same loader, the same chunk settings, and the same agent. Switching document sets means uploading different files, not editing Python.

Uploading more files adds them beside the ones already stored. Uploading the same filename again deletes only the chunks whose `source` is that filename, then re-indexes that file. Other filenames stay in Chroma.

## Stack

Nothing beyond this:

- Python, FastAPI, one Jinja page
- LangGraph `StateGraph`
- One local Chroma database on disk
- Chat model through OpenRouter, using `langchain_openai.ChatOpenAI`
- Embeddings with `HuggingFaceEmbeddings`, model `BAAI/bge-small-en-v1.5`, on this machine
- Chat model: `nvidia/nemotron-3-ultra-550b-a55b:free`
- OpenRouter base URL: `https://openrouter.ai/api/v1`
- `OPENROUTER_API_KEY` in `.env`
- No Postgres, no SQLite, no chat history in a database. `MemorySaver` only.

Out of scope: hybrid search, RRF, reranking, query decomposition, SQL, extra agents, and extra frameworks.

## How a question moves

`MessageState` is our own typed dict. It has one field, `messages`, annotated with `add_messages`. It is not LangGraph's prebuilt `MessagesState`.

One chat node prepends a short system message: answer from the uploaded documents, use the rag tool, and say you don't know when the documents do not have the answer. It binds `rag_tool` and calls the model.

`rag_tool` runs Chroma `similarity_search` (`k` = 4) and returns the source filename plus the passage. If nothing matches, it returns a short line that says so.

The graph is: start, chat node, `tools_condition`, a `ToolNode` that contains only `rag_tool`, then back to the chat node. Compile with `MemorySaver`.

There is one module-level `thread_id`. `ask(question)` invokes the graph with that id and returns the full thread as `{role, content}` pairs. Human messages and the final assistant text are included. Tool-call messages are skipped. Earlier turns stay in the thread, so a follow-up still has the previous question.

Each conversation in the sidebar is its own MemorySaver thread. Selecting a chat copies that chat's id into the module-level `thread_id`. A new upload replaces the active chat's thread id, so that conversation starts fresh, and leaves every other thread and every stored file alone.

## Phases

### Phase 1 — Project skeleton

Create `RAG-Generator` with pinned dependencies, `.env.example`, `.gitignore`, and `app/config.py`.

Config owns paths (`data/uploads`, `data/chroma`), the OpenRouter model name and base URL, allowed extensions, chunk size 1000, overlap 200, and retrieval k 4. The key is read from the environment. No document set is named here.

### Phase 2 — Models

`app/llm.py` caches one chat model and one embedding model.

- Chat: `ChatOpenAI`, temperature 0, OpenRouter base URL, model `nvidia/nemotron-3-ultra-550b-a55b:free`.
- Embeddings: `HuggingFaceEmbeddings`, `BAAI/bge-small-en-v1.5`, loaded on this machine the first time they are needed.

### Phase 3 — Shared document store

`app/store.py` is the runtime document pipeline, and it is the same pipeline for every file.

- Reject any extension other than `.txt`, `.md`, and `.pdf`.
- PDFs go through `PyPDFLoader`. Text and Markdown are read as UTF-8.
- Split with the configured chunk size and overlap.
- Write uploads under `data/uploads`.
- Add chunks to one shared Chroma collection under `data/chroma`.
- Set metadata `source` to the filename on every chunk.
- Re-index of the same filename deletes only those chunks, then adds the new ones.

Because the loader keys off the extension and the filename, a different document set does not require a code change.

### Phase 4 — Agent

`app/graph.py` defines `MessageState`, `rag_tool`, the chat node, and the compiled graph.

Also keep the in-memory chat list (id, title, thread id). This list is not a database. It dies with the process, same as `MemorySaver`. The module-level `thread_id` always points at the active chat. `ask` updates a "New chat" title from the first question, invokes the graph, and returns the visible thread.

### Phase 5 — API

`main.py`:

- `GET /` renders the page.
- `POST /api/upload` accepts one or more files, indexes them, resets the active chat's thread id, and returns the file names and chunk count.
- `POST /api/ask` accepts JSON `{question}` (and a chat id so the page can target a thread) and returns `{messages: the full thread}`.
- Chat list, create, select, rename, and delete stay in memory so the page can hold several conversations. Delete removes a thread from the list. It does not remove files from Chroma.

### Phase 6 — Page

One Jinja page, `templates/index.html`, laid out like a chat product.

- Conversations and the document list on the left, the thread on the right.
- New chat, switch chat, rename, delete.
- Text box. Enter sends. Shift+Enter inserts a newline.
- While a question is in flight, show exactly `Thinking....`.
- Render the assistant reply as simple Markdown: headings, lists, bold, italic, code.
- After each answer, replace the on-screen thread with the full `messages` list from the API.
- Say on the page that a new upload is added to the existing files and that the chat starts fresh.
- Empty, error, upload, and mobile states are handled in the page itself. No frontend framework.

### Phase 7 — README and install

`README.md` stays short: what the app is, one agent example, the mermaid graph, MemorySaver, the two model names, the file list, then venv, pip, `.env.example`, and uvicorn.

Then create a virtualenv and install `requirements.txt`.

## Done when

- A second, unrelated file can be uploaded without editing code, and questions about either file are answered from the indexed text.
- A question the files do not cover gets an "I don't know" style answer rather than a guessed one.
- A follow-up in the same chat still sees earlier turns.
- Another chat has its own thread. Uploading files starts a new thread for the active chat and leaves older Chroma chunks in place, except when the same filename is replaced.
