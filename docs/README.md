## System documentation — architecture + end-to-end flow

This document explains **what this repo does**, **how the agents cooperate**, and **which models/providers are invoked** at each step.

### Executive summary

The app is a periodic **deal-hunting agent system**:
- It scrapes a small set of RSS feeds (DealNews).
- An LLM picks the “best 5” deals and normalizes them into structured JSON (description, price, URL).
- An **ensemble price estimator** computes an “estimated true value” for each deal using:
  - a **fine-tuned specialist model** hosted on Modal
  - a **frontier LLM** grounded by **RAG** over a Chroma vector DB of products
  - (optional) a preprocessing LLM that rewrites/normalizes the product text first
- If the discount is large enough, it crafts a short notification message via an LLM and sends it via Pushover.
- It persists results in `memory.json` to avoid re-alerting on the same deals.

### High-level architecture (Mermaid)

```mermaid
flowchart TB
  UI[Gradio UI<br/>src/price_is_right.py] -->|startup + every 5 minutes| F[DealAgentFramework<br/>src/deal_agent_framework.py]
  F -->|loads/stores| M[memory.json]
  F -->|opens| C[(ChromaDB PersistentClient<br/>products_vectorstore/)]
  F -->|selects planner| P{PLANNER_MODE}

  P -->|workflow| W[PlanningAgent<br/>src/agents/planning_agent.py]
  P -->|autonomous tool-loop| A[AutonomousPlanningAgent<br/>src/agents/autonomous_planning_agent.py]

  W --> S[ScannerAgent<br/>src/agents/scanner_agent.py]
  A -->|tool: scan_the_internet_for_bargains| S
  S -->|RSS scrape + OpenAI structured output| OAI1[(OpenAI<br/>gpt-5-mini)]
  S --> D[DealSelection (5 deals)]

  W --> E[EnsembleAgent<br/>src/agents/ensemble_agent.py]
  A -->|tool: estimate_true_value| E
  E --> PP[Preprocessor (LiteLLM)<br/>ollama/llama3.2 default]
  E --> SA[SpecialistAgent (Modal)<br/>fine-tuned Llama 3.2 3B]
  E --> FA[FrontierAgent (RAG + OpenAI)<br/>gpt-5.1]
  FA -->|vector search| C
  FA -->|LLM price estimate| OAI2[(OpenAI<br/>gpt-5.1)]

  E --> OP[Opportunity: deal + estimate + discount]

  W -->|if discount > threshold| MSG[MessagingAgent<br/>src/agents/messaging_agent.py]
  A -->|tool: notify_user_of_deal| MSG
  MSG -->|craft copy| GROQ[(Groq via LiteLLM<br/>gpt-oss-20b)]
  MSG -->|send| PO[Pushover API]

  F -->|for visualization| TSNE[t-SNE + Plotly 3D plot]
  TSNE --> UI
```

### Runtime flow (what happens end-to-end)

#### 1) App boot

- `src/main.py`
  - loads `.env` (`python-dotenv`)
  - truncates `memory.json` to the first 2 entries (via `DealAgentFramework.reset_memory()`)
  - optionally builds the vector DB (`--build-vectordb`)
  - launches the Gradio UI (`App().run()`)

#### 2) UI triggers a run

- `src/price_is_right.py` sets a timer:
  - **startup**: run immediately when the UI loads
  - **periodic**: every 300 seconds (5 minutes)

The UI also shows:
- a table of found opportunities (price, estimate, discount, URL)
- a 3D plot of product embeddings from Chroma (t-SNE projection)
- streaming logs (reformatted ANSI → HTML)

#### 3) Framework chooses the planning strategy

- `src/deal_agent_framework.py` decides planner based on `PLANNER_MODE`:
  - **workflow**: `PlanningAgent`
  - **autonomous (default)**: `AutonomousPlanningAgent` (LLM tool-calling loop)

Both planners ultimately use the same capabilities:
- `ScannerAgent`: acquire and normalize deals
- `EnsembleAgent`: estimate “true value”
- `MessagingAgent`: notify user

#### 4) Deal acquisition (ScannerAgent)

- **Scrape**: `ScrapedDeal.fetch()` pulls entries from DealNews RSS feeds and then fetches each deal page to extract “Details” and “Features” HTML text (`feedparser`, `requests`, `beautifulsoup4`)
- **LLM selection**: `ScannerAgent` calls OpenAI **Structured Outputs** to produce a `DealSelection` (Pydantic schema)
  - **model**: `gpt-5-mini`
  - output: exactly 5 deals with a clear numeric price and a rewritten product description

#### 5) Price estimation (EnsembleAgent)

For each shortlisted deal, `EnsembleAgent.price()` does:

1) **Preprocess / rewrite** (optional but enabled in code)
   - `Preprocessor` uses `litellm.completion`
   - **default model**: `ollama/llama3.2` (local)
   - goal: normalize into Title/Category/Brand/Description/Details

2) **Specialist estimate (Modal)**
   - `SpecialistAgent` calls a Modal class `pricer-service.Pricer`
   - the Modal service loads a **fine-tuned** model:
     - **base**: `meta-llama/Llama-3.2-3B`
     - **fine-tune**: PEFT adapter `ed-donner/price-2025-11-28_18.47.07` (pinned revision in `src/modalApp/pricer_service*.py`)

3) **Frontier estimate (RAG + OpenAI)**
   - `FrontierAgent` embeds the rewritten text with SentenceTransformers (`all-MiniLM-L6-v2`)
   - retrieves 5 similar products from Chroma (`products_vectorstore`, collection `products`)
   - calls OpenAI with the deal description + retrieved examples to estimate a single price
     - **model**: `gpt-5.1`

4) **Combine**
   - current weighting: \(0.8 \cdot \text{frontier} + 0.2 \cdot \text{specialist}\)

#### 6) Notification (MessagingAgent)

- If a deal is compelling:
  - workflow mode checks `PlanningAgent.DEAL_THRESHOLD`
  - autonomous mode asks the planner LLM to call a “notify” tool once
- Message text is generated via `litellm` on Groq:
  - **model**: `groq/openai/gpt-oss-20b`
- Delivery uses **Pushover** HTTP API (`requests`)

#### 7) Persistence + visualization

- `memory.json` is appended with the best surfaced opportunity (if any)
- The 3D plot in the UI reads from Chroma and uses scikit-learn t-SNE
  - note: t-SNE default perplexity 30 ⇒ needs **at least 31 vectors**

### Vector DB build pipeline (products_vectorstore/)

The vector DB is built by `src/agents/rag_vectordb.py`:
- downloads an items dataset from HuggingFace (`datasets`)
  - default: `ed-donner/items_lite` (or `items_full`)
- embeds each item summary with SentenceTransformers `all-MiniLM-L6-v2`
- stores documents + embeddings + metadata `{category, price}` into ChromaDB (PersistentClient)

CLI entrypoints:

```bash
python3 src/main.py --build-vectordb
python3 src/main.py --build-vectordb --full-dataset
python3 src/main.py --build-vectordb --force-recreate-vectordb
```

### Key configuration knobs

- **Planner**: `PLANNER_MODE=workflow|autonomous`
- **Preprocessor model**: `PRICER_PREPROCESSOR_MODEL` (default `ollama/llama3.2`)
- **HF dataset user**: `HF_DATASET_USER` (default `ed-donner`)
- **Notification credentials**: `PUSHOVER_USER`, `PUSHOVER_TOKEN`
- **LLM credentials**: `OPENAI_API_KEY`, `GROQ_API_KEY`

### Where to look in code

- `src/main.py`: entrypoint and CLI flags (`--build-vectordb`, `--full-dataset`, `--force-recreate-vectordb`)
- `src/price_is_right.py`: Gradio UI + periodic execution + plots
- `src/deal_agent_framework.py`: planner selection + memory + vector DB + t-SNE plot data
- `src/agents/scanner_agent.py`: RSS → OpenAI structured deal selection (`gpt-5-mini`)
- `src/agents/ensemble_agent.py`: preprocessing + specialist + frontier combination
- `src/agents/frontier_agent.py`: RAG over Chroma + OpenAI price estimate (`gpt-5.1`)
- `src/agents/specialist_agent.py`: Modal RPC to fine-tuned Llama service
- `src/agents/messaging_agent.py`: Groq message crafting + Pushover send
- `src/agents/rag_vectordb.py`: builds the Chroma vector store from HF datasets

