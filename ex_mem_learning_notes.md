# ex_mem — Learning Notes & Architecture Planning

> Conversation with Claude — 2026-06-25

---

## Project Goal

Build **ex_mem**: an external memory system for LLMs that provides statefulness across sessions.  
Core mechanism: RAG-based memory retrieval with ultra-low latency, intelligent forgetting, and a graph layer on top of a vector DB.

**The problem being solved:** In a pure vector DB, semantically disconnected but relationally connected documents don't surface in queries. A graph layer solves this by traversing relationships even when embedding distance fails.

---

## The Core Problem

LLMs have no memory between sessions. Every call starts blank.  
Naive solution: store everything, load everything. Fails fast — too slow, too noisy.

### Why Pure Vector DB Fails

```
A: "User likes Python"
B: "User hates verbose code"
C: "User prefers concise one-liners"
```

Query: *"what coding style does the user prefer?"*  
Vector search finds A and C. **B is missed** — semantically distant but most informative.  
Graph fix: B and C are connected by relationship ("coding style preference"), traversal pulls B in.

---

## Learning Phases Roadmap

| Phase | What to Build | What to Learn |
|-------|--------------|---------------|
| 1 | File-based memory + Ollama | Baseline, serialization, latency measurement |
| 2 | Vector DB (ChromaDB) + embeddings | Semantic search, recall vs precision |
| 3 | Hybrid search (BM25 + vector) | When keywords beat semantics |
| 4 | Graph layer | Relationship traversal, connected doc problem |
| 5 | Forgetting | Recency, importance, memory decay |

Each phase keeps the previous — building up, not replacing.

---

## Memory Schema Design

### Fields Decided

```json
{
  "id": "uuid",
  "text": "the memory fact",
  "timestamp": "ISO8601",
  "importance_score": 0.0,
  "topic": "skills | preferences | domain | ...",
  "extraction_confidence": "certainly | likely | maybe | guessing",
  "user_certainty": "certainly | likely | maybe | guessing",
  "access_count": 0,
  "last_accessed": "ISO8601",
  "source_turn": "conversation turn reference"
}
```

### Confidence Levels — Two Separate Signals

- `extraction_confidence`: how sure is the LLM that it correctly *extracted* this fact (model uncertainty)
- `user_certainty`: how sure is the *user* about this fact (domain uncertainty)

Example: user says "I think I prefer Python" → user_certainty: maybe, extraction_confidence: certainly

### Conflict Resolution

Timestamp decides which wins, but **timestamp alone is weak**.  
Detection step is required first:
1. Embed the new memory
2. Similarity search against existing memories
3. If similarity > threshold AND they contradict → apply timestamp logic (keep newer)
4. If no conflict → insert

---

## Phase 2 Architecture (Starting Point — skipping Phase 1 due to prior RAG experience)

### Write Path

```
User input
    ↓
Ollama (extraction prompt) → structured memory JSON
    ↓
Embed the memory text (nomic-embed-text)
    ↓
Similarity search against existing memories
    ↓
  If conflict found → compare timestamps → keep newer, soft-delete older
  If no conflict → insert
    ↓
ChromaDB (vector + metadata stored together)
```

### Read Path

```
User query
    ↓
Embed the query
    ↓
ChromaDB similarity search (with metadata filters if needed)
    ↓
Top-k memories injected into system prompt
    ↓
Ollama generates response
```

### Tools

- **ChromaDB** — vector store + metadata + persistence (preferred over FAISS for this use case)
- **Ollama** — embeddings (`nomic-embed-text`) + generation (`llama3.2` or similar)
- **Pydantic** — schema validation of extracted memories (don't trust raw LLM JSON)

### The Hardest Part — Memory Extraction Prompt

Turn this:
> *"Yeah I've been using Python for about 3 years, mostly for data work, I hate writing boilerplate"*

Into this:
```json
[
  {"text": "Uses Python for 3 years", "topic": "skills", "user_certainty": "certainly"},
  {"text": "Focuses on data work with Python", "topic": "domain", "user_certainty": "likely"},
  {"text": "Dislikes boilerplate code", "topic": "preferences", "user_certainty": "certainly"}
]
```

**Key questions before writing the prompt:**
1. What is NOT worth storing? (greetings, filler, questions user asks vs states)
2. Granularity: one big memory or many small ones? (small = better precision, harder to extract)

---

## Competitor Analysis

### ChatGPT Memory (OpenAI)

- Stores plain text facts
- Extracts with LLM call after each conversation
- Retrieves with embedding similarity → injects top-k into system prompt
- LLM decides UPDATE vs ADD on conflict
- User can view/delete memories

**Gap:** Flat list, no relationships, no forgetting. Memory grows forever.

---

### Mem0

Core loop:
```
New conversation turn
    ↓
LLM extracts candidate memories
    ↓
Vector search against existing memories
    ↓
LLM looks at (new memory + similar existing) and decides:
    ADD    → completely new information
    UPDATE → refines an existing memory
    DELETE → contradicts and replaces
    NOOP   → already known, skip
    ↓
Execute the operation
```

**Stack:** Qdrant (vector) + Neo4j (graph, optional/paid)

**Key insight:** LLM does conflict reasoning, not rule-based logic.

**Gap:** Extra LLM call on every write adds latency. Graph underutilized in open source. No forgetting.

---

### Zep — Most Sophisticated

Solves **temporal validity** — something others don't:

```
Instead of overwriting:
  "User is Python beginner"  →  "User is Python expert"

Zep does:
  "User is Python beginner"   valid: [Jan → March]
  "User is Python expert"     valid: [March → now]
```

Every fact has `valid_from` and `valid_to`. Nothing deleted — it expires.  
Extracts *entities* and *relationships*, not just flat facts.

```
Entity: User
Relationship: [PREFERS] → Python         (valid: March → now)
Relationship: [WORKS_IN] → Data Eng      (valid: Jan → now)
```

**Gap:** Complex infra, hard to self-host, expensive.

---

### Letta / MemGPT — Inverted Approach

The LLM manages its own memory via function calls:  
`read_from_archival_memory()`, `write_to_archival_memory()`, `search_recall_memory()`

**Smart:** Model knows what it needs. **Slow:** Every memory op = an LLM call with latency.

---

## Competitor Gap Table

| System | Core Gap |
|--------|----------|
| ChatGPT | Flat list, no relationships, no forgetting |
| Mem0 | Extra LLM call per write hurts latency, graph underused |
| Zep | Complex infra, hard to self-host, expensive |
| Letta | Slow — every memory op = an LLM call |

---

## ex_mem Differentiation (To Be Defined)

**Potential advantages:**
1. No LLM call on write conflict check — use embedding similarity + simple rule (not another LLM call like Mem0) to decide ADD/UPDATE/DELETE
2. Zep's temporal validity idea without the full knowledge graph complexity
3. Forgetting as a first-class feature, not an afterthought
4. Ultra-low latency as a core design constraint throughout

**Open question (to answer before next session):**  
*What is ex_mem doing that Mem0 doesn't?*  
That answer becomes the project's thesis and shapes every architectural decision.

---

## Next Steps

- [ ] Define ex_mem's differentiation thesis vs Mem0
- [ ] Write the memory extraction prompt (just the text, no code yet)
- [ ] Set up Ollama locally with `nomic-embed-text` + a generation model
- [ ] Design the ChromaDB collection schema
- [ ] Build Phase 2 write path
- [ ] Instrument latency measurement from day one (retrieval separate from LLM inference)
