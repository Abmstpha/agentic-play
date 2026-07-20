# Part 2 — Block 1: Advanced RAG & MCP

Course + lab for Block 1 (Aivancity PGE5 · Agentic AI).

## Contents

| File | What it is |
|---|---|
| `COURSE_NOTES.md` | Full course notes parsed from the Block 1 slide deck (PDF not committed) |
| `lab_B1_advanced_rag.ipynb` | Completed lab, executed end to end (all outputs included) |
| `mcp_server.py` | Deliverable: 3-tool MCP server (`web_search`, `recall_memory`, `store_finding`) — generated and verified by the notebook |
| `llm_helpers.py` | Course helper library (multi-provider LLM client + mock) |
| `requirements.txt` | Dependencies for this part |

## What the lab covers

1. **Baseline** — pure-Python TF-IDF retriever; the hit@3 / MRR metrics are defined up front and all retrievers are compared in section 6.
2. **Fix 1: parent-child chunking** — retrieve small children, return large parents.
3. **Fix 2: hybrid search** — BM25 + TF-IDF fused with Reciprocal Rank Fusion.
4. **Fix 3: cross-encoder reranking** — joint query×document scoring of the candidates.
5. **RAGAS** — hit@3 / MRR proxy offline, plus a **real RAGAS run** on the 5-question test set. The judge is printed in the cell output as evidence (`model='mistral-large-latest' base_url='https://api.mistral.ai/v1'`); all 15 judge calls completed (sequential + retry backoff, no rate-limit failures). Production-pipeline scores:

   | Metric | Score |
   |---|---|
   | context_recall | **1.00** |
   | context_precision | **1.00** |
   | faithfulness | **1.00** |

   Two honest caveats: `answer_relevancy` is skipped (Mistral exposes no embeddings endpoint), and cell 6b scores the top retrieved chunk as the "answer", which makes faithfulness near-trivial there — on a toy 12-doc corpus these 1.00s mainly confirm the pipeline retrieves the right documents.
6. **Retriever as an agent tool** — run live on Mistral, with written answers to the analysis questions based on the actual outputs. Worth noting: the model's answer added 7 countries that are not in the retrieved context (details in the Q3 answer) — a concrete faithfulness failure the 9.1 filter is meant to catch.
7. **MCP server** — built, launched as a real subprocess and verified over the MCP protocol: `list_tools` ✓, `recall_memory` ✓, store→recall round-trip ✓, `web_search` ✓ (graceful no-results string — this verifies the error-handling contract, not live search results). **Deliverable 1b: 3/3 tools passing.**
8. **Exercises 9.1 & 9.2** — groundedness filter (warns below 0.5) and metadata-filtered hybrid retrieval, both implemented and demonstrated.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab lab_B1_advanced_rag.ipynb
```

The submitted notebook was executed **live end to end** (`LLM_PROVIDER=mistral`,
`mistral-large-latest`); every cell completed without exceptions. Without a key it also runs fully **offline**
(deterministic `MockLLMClient`) — to go live, create a `.env` file in this folder (it is
gitignored) with your key and `LLM_PROVIDER` (`openai | mistral | google | anthropic`).

> Free-tier keys rate-limit at ~1 req/s; `llm_helpers.py` was extended with a small
> retry-with-backoff on 429 so the full notebook survives a live run. For the real-RAGAS
> cell, point the judge at Mistral's OpenAI-compatible endpoint:
> `OPENAI_BASE_URL=https://api.mistral.ai/v1 OPENAI_API_KEY=$MISTRAL_API_KEY LLM_MODEL=mistral-large-latest`.

Inspect the MCP server visually (needs Node):

```bash
npx @modelcontextprotocol/inspector python mcp_server.py
```
