# Document chat application
_Exported on 10/1/2026 at 00:14:07 GMT+5:30 from Cursor (3.22.12)_

---

**User**

Build a small document chat app I can run locally. I upload .txt, .md, or .pdf files in the browser, then ask questions in the same conversation. Answers must come from those files. If the files do not contain the answer, the agent says it does not know.
Stack, and do not add anything else:
- Python, FastAPI, one Jinja page
- LangGraph StateGraph
- One local Chroma database on disk
- Chat model through OpenRouter, using langchain_openai.ChatOpenAI
- Embeddings with HuggingFaceEmbeddings, model BAAI/bge-small-en-v1.5, running on this machine
- Chat model: nvidia/nemotron-3-ultra-550b-a55b:free
- OpenRouter base URL: https://openrouter.ai/api/v1
- OPENROUTER_API_KEY I will set in the .env
- No Postgres, no SQLite, no chat history in a database use MemorySaver
Pin these versions in requirements.txt and use others whichever are required  and ensure compatibility with these:
langgraph==1.2.2
langchain==1.3.2
langchain-openai==1.4.3
langchain-community==0.4.2
langchain-chroma==1.1.0
langchain-huggingface==1.2.2
sentence-transformers==6.1.0
langchain-text-splitters==1.1.2
chromadb==1.5.7
pypdf==6.7.4
Files:
- main.py — FastAPI app
- app/config.py — paths, model name, allowed extensions, chunk size 1000, overlap 200, retrieval k 4
- app/llm.py — cached chat model (temperature 0) and cached embedding model
- app/store.py — read files, chunk them, add them to one shared Chroma collection
- app/graph.py — the agent
- templates/index.html — the page
- README.md — short
Agent:
- Define our own MessageState. It has one field, messages, annotated with add_messages. Do not use LangGraph's prebuilt MessagesState.
- One chat node. It adds a short system message: answer from the uploaded documents, use the rag tool, and say you don't know when the documents do not have the answer. Then bind that tool and call the model.
- rag_tool runs Chroma similarity_search and returns the source filename plus the passage text. If nothing matches, return a short line that says so.
- Graph: start, chat node, tools_condition, a ToolNode with only rag_tool, then back to the chat node. Compile with MemorySaver.
- Keep one module-level thread id. ask(question) invokes the graph with that id and returns the full thread as a list of {role, content}, including earlier user and assistant turns. Include human messages and the final assistant text. Skip tool-call messages.
- A new upload starts a new thread id. It does not delete other files from Chroma.
- Uploading more files adds them beside the ones already stored. Uploading the same filename again deletes only the chunks whose source is that filename, then re-indexes that file.
- Store uploads under data/uploads and Chroma under data/chroma. Tag every chunk with metadata source set to the filename.
- PDFs go through PyPDFLoader. txt and md are read as UTF-8. Reject any other extension.
API and page:
- GET / renders the page.
- POST /api/upload accepts one or more files, indexes them, resets the chat thread, and returns the file names and chunk count.
- POST /api/ask accepts JSON {question} and returns {messages: the full thread}.
- The page is a simple chat: files on the left, messages on the right, a text box. Enter sends, Shift+Enter inserts a newline.
- While the request is in flight, show exactly: Thinking....
- Render the assistant reply as simple Markdown (headings, lists, bold, italic, code).
- After each answer, replace the on-screen thread with the full messages list from the API, not only the latest pair.
- Say on the page that a new upload is added to the existing files and that the chat starts fresh.
README:
- Call it a small chat app for asking questions about uploaded PDFs and text files.
- Explain the agent with one concrete example, then a short mermaid diagram: Start, chat node, rag tool, back to the chat node, End.
- State that MemorySaver keeps the thread until the process stops, and that a new upload starts a new thread without deleting older files from Chroma.
- Name the OpenRouter model and the Hugging Face embedding model, and say an OpenRouter key is required.
- List the files in one short group, then the venv, pip install, copy .env.example, and uvicorn main:app --reload.
- Do not add curl examples, sample-file instructions, or a description of the page layout.
Leave these out: hybrid search, RRF, reranking, query decomposition, SQL, extra agents, and extra frameworks.
When you are done, install the requirements.

One more thing to keep in mind is put specific attention to the problem sttement points especially 4th point:
Problem Statement
Build a RAG Generator that:
• Accepts documents at runtime
• Creates a RAG application over those documents
• Allows users to ask questions and receive grounded answers
• Works with different document sets without code changes


Before start working prepare a md file in which it will have the full plan of the developement in phases. Make the UI very robust and look like chat application similar to chatgpt. It should suppport multiple chats, threads etc. 

You can start working inside the RAG-Generator folder

---

**Cursor**

I'll start by looking at the workspace and the assignment notes so the plan matches the stack you specified, then I'll write the phased plan in `RAG-Generator` before building the app.

The assignment needs a runtime document set, one shared Chroma index, and separate chat threads. I’ll write that plan in `RAG-Generator` first, then build the app to match it.

The plan is in `RAG-Generator/PLAN.md`. Next I’ll add the agent, the API, and the chat page.

The install is running. I’m tightening a few spots that would block the page or hide a failed search while that finishes.

Requirements are installed. Next I’ll check indexing, chat threads, and the API against the problem statement.

The page and empty-library path check out. I’ll index two different files next and confirm a new upload only replaces the matching filename.

The live model answered from the handbook, and a follow-up stayed in the same thread. Nvidia’s free endpoint also returns overload errors as a `ValueError`, so I’m making that message readable and keeping a failed question from being duplicated in the box.

The app is in `RAG-Generator`. The development plan is in `PLAN.md`, and the requirements are installed in `.venv`.

It is a local document chat. You upload `.txt`, `.md`, or `.pdf` files, then ask questions in that conversation. The same code indexes whatever you upload, so a different document set does not require a code change. Answers come from those files. If they do not contain the answer, the agent is instructed to say it does not know.

The page is a chat with a conversation list and the document library on the left, and the thread on the right. You can keep several chats. Each one is its own thread in `MemorySaver`. A new upload is added beside the files already stored, and that chat starts fresh. Uploading the same filename again replaces only that file. Older files stay in the Chroma database on disk. Chat history lasts until you stop the process.

I indexed text, Markdown, and PDF into one Chroma collection, checked that a same-name upload replaces only that file, and checked that a second chat keeps its own thread. With the OpenRouter model, a handbook that says full-time staff get 18 paid leave days produced: “full-time staff get **18 paid leave days each year**.” A follow-up in the same thread still had the earlier question. The free Nvidia model can also come back busy; the page then says to try again in a moment, instead of leaving `Thinking....` on screen.

From `RAG-Generator`:

```bash
source .venv/bin/activate
cp .env.example .env   # if you have not added OPENROUTER_API_KEY yet
uvicorn main:app --reload
```

Open http://127.0.0.1:8000. The first upload downloads `BAAI/bge-small-en-v1.5` onto this machine.

---

**User**

<timestamp>Wednesday, Sep 30, 2026, 11:55 PM (UTC+5:30)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 991562
title: Create venv and install requirements
tool_call_id: call-49d98f25-92f6-4dab-b1cb-4770eed98bc4-30
fc_76b6b1f8-054b-9b56-8528-b14ded0f9147_1
output_path: /Users/shivamgupta/.cursor/projects/Users-shivamgupta-Desktop-Study-AI-AI-Agents-LangGraph/terminals/991562.txt
</task>
</system_notification>

<timestamp>Wednesday, Sep 30, 2026, 11:55 PM (UTC+5:30)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 991563
title: Test API, page, and empty retrieval
tool_call_id: call-2a8147d3-a37f-4506-a463-aacaec3882b1-44
fc_40c480bf-bb34-98b1-9128-24208866c400_0
output_path: /Users/shivamgupta/.cursor/projects/Users-shivamgupta-Desktop-Study-AI-AI-Agents-LangGraph/terminals/991563.txt
</task>
</system_notification>

<timestamp>Wednesday, Sep 30, 2026, 11:55 PM (UTC+5:30)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 991564
title: Index files and test chats, replace, PDF
tool_call_id: call-fc2f312c-d008-4f51-a290-cb20697f274f-46
fc_fff4c768-7c16-9a46-abb8-f50dc7add262_0
output_path: /Users/shivamgupta/.cursor/projects/Users-shivamgupta-Desktop-Study-AI-AI-Agents-LangGraph/terminals/991564.txt
</task>
</system_notification>

<timestamp>Wednesday, Sep 30, 2026, 11:55 PM (UTC+5:30)</timestamp>
<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: aborted
task_id: 991565
title: Start RAG Generator on port 8000
tool_call_id: call-2c1742cf-aae2-437a-9137-16b90ac2d0cd-53
fc_f2f57fc9-4200-9d53-8214-a7ce557e832f_1
output_path: /Users/shivamgupta/.cursor/projects/Users-shivamgupta-Desktop-Study-AI-AI-Agents-LangGraph/terminals/991565.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed). If there's no follow-ups needed, don't explicitly say that.</user_query>

---

**Cursor**

The install and the document tests finished successfully. The local server on port 8000 was stopped, so you can start it yourself when you are ready.

From `RAG-Generator`:

```bash
source .venv/bin/activate
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000. Put `OPENROUTER_API_KEY` in `.env` if it is not there yet.

---

**User**

I want you to check that how you are supporting multiple documents. By not changing the backend code. So, I do not see anything of letting me support on the UI. So please check that rapidly. And make changes according to that. And do not take much time. I only have about ten minutes left, so I want you to verify the changes you done you do, and finally test it and complete it within five to six minutes. Do not change much just you can just change a bit. And one more thing I see, that whenever I upload a new document, the chat gets reload. And it is removed. So, the old chat is not valid anymore. So, I want that even on uploading the new document the old chat is still there. It is not refreshed. So, please make sure you make these changes fast and within five to six minutes. And also verify them.

---

**Cursor**

Uploading a file currently starts a new thread, which drops the conversation. I’ll keep that chat in place and make it obvious on the page that extra documents are added to the same library.

Uploading a file no longer clears the conversation. The new file is added next to the ones already stored, and the messages stay in the same chat.

The document list now says you can add as many `.txt`, `.md`, or `.pdf` files as you want, and it shows how many are stored. Both of your invoices are still there. I checked this on the running app: after a question about the invoices, uploading another file left that question and answer in place, with all three files listed.

Refresh the page once so the updated note appears. The reload that picked up this change cleared the in-memory thread that one time. Uploads after that keep the chat.

---

**User**

add a detailed readme for the same add the agent state, models used mermaid diagram for agent workflow.

Also I have listed screenshots of the agent, please add them in the end of readme. make readme medium length

---

**Cursor**

I'll look at the screenshots and the current README, then expand it with the agent state, models, and workflow diagram.

There’s one screenshot in `screenshots`. I’ll write a medium README that covers the agent state, the models, the workflow diagram, and that image at the end.

The README in `RAG-Generator` is updated. It now covers `MessageState` (the single `messages` field and `add_messages`), the OpenRouter chat model and the local Hugging Face embedding model, and a mermaid diagram of Start, the chat node, the rag tool, and back to the chat node.

The screenshot from `screenshots` is at the end. It shows the AWS invoice question, with both PDFs still listed and the chat still open after the upload.

