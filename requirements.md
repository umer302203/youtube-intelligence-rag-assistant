# Dependency Requirements

This document explains the runtime dependencies declared in `requirements.txt`. The list is derived from the imports in `ytbot.py`; it is not a generic AI template.

| Package | Why this project uses it | Imported capability |
|---|---|---|
| `gradio` | Builds the browser UI, tabs, textboxes, buttons, callbacks, theme, CSS, and temporary share endpoint. | `import gradio as gr` |
| `youtube-transcript-api` | Lists and fetches caption tracks for a YouTube video ID. | `YouTubeTranscriptApi` |
| `langchain` | Provides prompt and chain compatibility used by the current implementation. | `LLMChain`, `PromptTemplate`, text splitter compatibility |
| `langchain-text-splitters` | Provides the maintained recursive character splitter implementation used for RAG chunking. | `RecursiveCharacterTextSplitter` |
| `langchain-community` | Provides the FAISS vector-store integration. | `FAISS` |
| `langchain-ibm` | Provides LangChain wrappers for IBM watsonx.ai generation and embeddings. | `WatsonxLLM`, `WatsonxEmbeddings` |
| `ibm-watsonx-ai` | Provides IBM watsonx.ai credentials, client-related SDK objects, model enums, decoding enums, and generation parameter names. | `APIClient`, `Credentials`, `ModelTypes`, `GenParams`, `DecodingMethods` |
| `faiss-cpu` | Performs local vector similarity search without requiring a GPU. | FAISS index used through LangChain |

## Installation contract

Use a virtual environment and install the executable manifest:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The project should be tested on the Python version selected for deployment. FAISS and IBM SDK compatibility can vary across operating systems and Python versions, so a clean installation should be verified before deployment.

## Authentication contract

The source code does not commit an API key. IBM watsonx.ai credentials must be configured in the runtime environment according to the IBM SDK's supported authentication mechanism. The hard-coded region URL is `https://us-south.ml.cloud.ibm.com`, and the current source uses project ID `skills-network`; both should be reviewed before deploying to a personal IBM Cloud project.

## Dependency maintenance

The version ranges are intentionally bounded at the major-version level. For reproducible production builds, generate a lock file or a fully pinned deployment manifest after testing a known-good environment. Upgrade LangChain packages together rather than independently because their integration and response contracts are closely related.
