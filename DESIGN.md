# DESIGN.md - Multi-Agent Tool-Use Conversation Generator

---

## Table of Contents

1. [Architecture & Decisions](#1-architecture--decisions)
2. [Data Model - Tool Registry](#2-data-model--tool-registry)
3. [Tool Graph Construction & Sampling](#3-tool-graph-construction--sampling)
4. [Offline Execution Model](#4-offline-execution-model)
5. [Multi-Agent Conversation Generator](#5-multi-agent-conversation-generator)
6. [Quality Evaluation Pipeline](#6-quality-evaluation-pipeline)
7. [Context Management Design](#7-context-management-design)
8. [Prompt Design](#8-prompt-design)
9. [Diversity & Quality Analysis](#9-diversity--quality-analysis)
10. [Output Schema](#10-output-schema)
11. [What I Would Do Differently at Scale](#11-what-i-would-do-differently-at-scale)
12. [Observed Failure Modes & Fixes](#12-observed-failure-modes--fixes)

---

## 1. Architecture & Decisions

### 1.1 System Overview

The system is a five-stage pipeline that takes raw ToolBench JSON and produces a JSONL dataset of multi-turn, multi-tool conversations with quality scores.

```
ToolBench JSON
     │
     ▼
┌─────────────────┐
│  Tool Registry  │  Parse & normalize raw API schemas
└────────┬────────┘
         │  List[Tool]
         ▼
┌─────────────────┐     ┌─────────────────────┐
│   Tool Graph    │◄────│  Coverage Tracker   │
│   (networkx)    │     │  (corpus steering)  │
└────────┬────────┘     └─────────────────────┘
         │  ToolChain (sampled)
         ▼
┌──────────────────────────────────────────────────────┐
│                    Orchestrator                      │
│                                                      │
│  ┌──────────┐  ConversationPlan  ┌────────────────┐  │
│  │ Planner  │──────────────────► │ Dialogue Loop  │  │
│  │  Agent   │                   │                │   │
│  └──────────┘                   │  User Agent    │   │
│                                 │      ↕         │   │
│                                 │  Assistant     │   │
│                                 │  Agent         │   │
│                                 │      │         │   │
│                                 │  [tool_call]   │   │
│                                 │      │         │   │
│                                 │  Execution     │   │
│                                 │  Session       │   │
│                                 └────────────────┘   │
└─────────────────────┬────────────────────────────────┘
                      │ Conversation
                      ▼
               ┌─────────────┐
               │ Judge Agent │  Score on 3 dimensions
               └──────┬──────┘
                      │
            pass ─────┴───── fail
             │                 │
        Write JSONL       Repair Loop
                         (max 3 attempts)
```

### 1.2 Key Design Decisions

**Sequential agent pipeline over concurrent orchestration.** Conversation turns are inherently sequential - each message depends on all prior messages. Concurrency makes sense at the corpus level (multiple conversations in parallel) but not within a single conversation. A sequential loop is simpler to debug and reason about.

**No shared mutable state between conversations.** Each conversation gets its own `ExecutionSession` and `ConversationContext`. The only shared state is the `CoverageTracker`, which is explicitly designed as a write-append ledger, not a cache.

**Structured output at the seams.** The Planner Agent and Judge Agent both use JSON-mode structured output. These are the two places where downstream components need to programmatically parse LLM outputs. The User and Assistant agents use free text for their conversational roles - structured output there would hurt naturalness.

**Graph-first sampling, always.** The hard requirement is that the generator must use the graph sampler, not a hardcoded list. Every conversation starts with a graph walk. This enforces that tool chains are relationally grounded, not arbitrary.

### 1.3 Component Responsibilities

| Component | Responsibility | Output |
|-----------|---------------|--------|
| `registry.Loader` | Parse raw ToolBench JSON → clean models | `List[Tool]` |
| `graph.Builder` | Construct Tool Graph from registry | `ToolGraph` (networkx DiGraph) |
| `graph.Sampler` | Sample tool chains with constraints | `ToolChain` |
| `graph.CoverageTracker` | Track what's been generated; compute steering weights | Weight adjustments |
| `executor.MockGenerator` | Generate mock tool responses | `Dict[str, Any]` |
| `executor.ExecutionSession` | Maintain session state for chaining | `Dict` (running state) |
| `agents.Planner` | Generate structured conversation plan | `ConversationPlan` |
| `agents.UserSimulator` | Simulate user turns | `str` |
| `agents.Assistant` | Generate assistant turns + tool calls | `AssistantTurn` |
| `agents.Judge` | Score and diagnose conversations | `JudgeResult` |
| `agents.Orchestrator` | Run the dialogue loop + repair | `Conversation` |
| `output.Writer` | Serialize conversations to JSONL | `.jsonl` file |

---

## 2. Data Model - Tool Registry

### 2.1 Problem

ToolBench ships thousands of API definitions with inconsistent structure: some have detailed response schemas, some have `null` parameter types, some mix required/optional in the description field rather than a dedicated key. A fragile loader breaks on these; a robust loader normalizes gracefully.

### 2.2 Internal Data Model

```python
@dataclass
class Parameter:
    name: str
    type: Literal["string", "number", "boolean", "array", "object", "unknown"]
    description: str
    required: bool
    default: Optional[Any] = None
    enum: Optional[List[Any]] = None

@dataclass
class Tool:
    id: str                          # "{category}/{api_name}/{endpoint_name}"
    category: str                    # ToolBench top-level category
    api_name: str
    endpoint_name: str
    description: str
    parameters: List[Parameter]
    required_params: List[str]       # derived from parameters for fast lookup
    response_schema: Optional[Dict]  # may be None; handled downstream
    raw: Dict                        # original JSON preserved for debugging
```

### 2.3 Normalization Rules

| Raw field state | Normalization action |
|----------------|---------------------|
| `type: null` or missing | Map to `"unknown"` - downstream mock generator handles this |
| `required` field absent | Infer from description: if "required" appears in description text, mark required |
| No `response_schema` | Set `None`; mock generator falls back to LLM generation |
| Duplicate endpoint names | Deduplicate by appending `_v2`, `_v3`; log warning |
| Missing description | Use `api_name + endpoint_name` as fallback description |
| Parameters as string (not list) | Parse comma-separated names, type defaults to `"unknown"` |

### 2.4 Why This Model

- **Canonical ID format** (`category/api/endpoint`): readable, filterable by prefix, unique within the registry. Avoids integer IDs that are meaningless outside the registry.
- **`required_params` as derived field**: redundant with `parameters` but avoids iterating the list in hot paths (graph edge detection, mock generation).
- **`raw` preserved**: when debugging why a conversation went wrong, you want the original schema, not just the normalized version.
- **`type: "unknown"`** over `None`: makes downstream code handle the case explicitly rather than crash on `None`.

---

## 3. Tool Graph Construction & Sampling

### 3.1 Graph Schema

**Nodes** represent tools. Each node stores:
- `tool_id: str`
- `category: str`
- `description_embedding: Optional[np.ndarray]` - computed lazily, used for diversity metrics
- `use_count: int` - how many times this tool has appeared in generated conversations

**Edges** represent a "can chain into" relationship: tool A → tool B means "A produces output that B can consume as input." Each edge stores:
- `matched_field: str` - the field name that connects them
- `confidence: float` - 0.0–1.0
- `pattern: Literal["sequential", "parallel"]`

### 3.2 Edge Detection Strategy

The central challenge: how to detect output→input compatibility without running the tools. Three methods, applied in priority order:

**Method 1 - Exact name match (confidence: 0.9)**
Tool A's response schema contains field `X`; tool B has a required parameter named `X` with a compatible type. Example: `hotels/search` response has `hotel_id`, `hotels/book` requires `hotel_id`.

**Method 2 - Semantic name match (confidence: 0.6)**
Tool A's response has a field named `id` or `result_id`, and tool B requires a parameter whose name contains the source tool's domain keyword. Example: `hotels/search` response has `id` → `hotels/book` requires `hotel_id` (both in `hotels` category).

**Method 3 - Same-category co-occurrence (confidence: 0.3)**
Tools in the same ToolBench category are connected as potential parallel-use candidates. These are weak edges used to enable same-domain multi-tool conversations even when output→input chaining isn't possible.

**Why not LLM-based edge detection?** With O(n²) tool pairs, LLM-based detection at build time is prohibitively expensive for thousands of tools. Name matching is O(n²) comparisons but each comparison is microseconds. The tradeoff: we miss chains that require semantic understanding (e.g., `hotel_result` feeding `accommodation_id`) but we never generate false chains. I consider false edges worse than missed edges because false edges produce hallucinated conversations.

### 3.3 Constrained Sampling

```python
@dataclass
class SamplingConstraint:
    min_steps: int = 2
    max_steps: int = 6
    required_domains: List[str] = field(default_factory=list)
    required_tool_ids: List[str] = field(default_factory=list)
    forbidden_tool_ids: Set[str] = field(default_factory=set)
    pattern: Literal["sequential", "parallel", "mixed"] = "sequential"
    length_distribution: Dict[int, float] = field(
        default_factory=lambda: {2: 0.20, 3: 0.30, 4: 0.30, 5: 0.15, 6: 0.05}
    )
```

The sampler runs a **weighted random walk**:
1. Pick a start node: weight = `1 / (1 + log(1 + use_count))` (inverse frequency, from CoverageTracker)
2. At each step: filter candidate next nodes by constraint; weight edges by confidence × inverse_use_count
3. Backtrack if no valid next node exists (max 10 attempts before relaxing constraints)
4. Return the walk as a `ToolChain`

**Constraint relaxation order** (when no valid walk is found): first relax `required_domains`, then `min_steps`, then log a warning and return what we have.

### 3.4 Parallel Pattern Sampling

For parallel patterns, the sampler:
1. Picks a "fan-out" node (tool with 2+ outgoing edges in the same domain)
2. Samples 2–3 siblings that can run in parallel (same required inputs, no dependency on each other)
3. Picks a "fan-in" node that consumes outputs from all siblings

This is an approximation - true parallelism detection requires runtime analysis. We document it as "structurally parallel" in the conversation metadata.

---

## 4. Offline Execution Model

### 4.1 The Chaining Problem

The central failure mode in multi-step tool use: if step 1 returns `{"hotel_id": "htl_881"}` and step 2 asks for `hotel_id`, the LLM assistant tends to hallucinate a plausible-sounding ID rather than reference the actual returned value. The execution model must prevent this.

### 4.2 Execution Session

```python
class ExecutionSession:
    """Maintains state across tool calls in a single conversation."""
    
    state: Dict[str, Any]  # extracted_field_name -> value
    history: List[ToolCallRecord]
    
    def execute(self, tool: Tool, input_args: Dict) -> Dict:
        # 1. Resolve arguments: prefer state values over LLM-generated args
        resolved = self._resolve_args(tool, input_args)
        
        # 2. Generate mock response
        response = self.mock_generator.generate(tool, resolved)
        
        # 3. Extract key fields into state
        self.state.update(self._extract_refs(response))
        
        # 4. Record
        self.history.append(ToolCallRecord(tool.id, resolved, response))
        
        return response
    
    def _resolve_args(self, tool: Tool, args: Dict) -> Dict:
        """Replace any arg whose name is in state with the actual state value."""
        resolved = {}
        for param in tool.parameters:
            if param.name in args:
                # If state has this value, use it (grounding); else use LLM value
                resolved[param.name] = self.state.get(param.name, args[param.name])
        return resolved
    
    def _extract_refs(self, response: Dict) -> Dict:
        """Extract ID-like fields for future use."""
        refs = {}
        for key, value in self._flatten(response).items():
            if any(key.endswith(suffix) for suffix in 
                   ("_id", "_key", "_token", "_ref", "_code", "_number")):
                refs[key] = value
        return refs
```

### 4.3 Mock Generator

Two strategies, chosen based on whether a response schema exists:

**Schema-derived (preferred)**
Walk the response schema, generate type-conformant values using `faker`:
- `string` → `faker.word()` or `faker.sentence()` based on field name heuristics
- `number` → random float/int in plausible range
- `boolean` → `random.choice([True, False])`
- `array` → 1–5 elements, each schema-derived
- IDs → `faker.uuid4()` or domain-prefixed `{domain}_{random_hex[:6]}`

**LLM-generated fallback (when no schema)**
A single LLM call with the tool description and resolved input args, asking for a plausible JSON response. The response is parsed and validated; if it fails, we fall back to a generic `{"status": "success", "result": null}` rather than crashing.

**Tradeoff**: LLM mocks are more semantically coherent (better tool output descriptions) but cost ~5–10× more per conversation. At 100 conversations × 4 tool calls average, that's potentially 400 extra LLM calls just for mocks. I use schema-derived mocks as the primary path and LLM mocks only when no schema exists.

---

## 5. Multi-Agent Conversation Generator

### 5.1 Agent Roles

#### Planner Agent (structured output)

The Planner receives the sampled tool chain and produces a structured conversation plan. This is the one agent that **must** use structured output - the Orchestrator parses its output programmatically to inject disambiguation turns at the right moments.

**Output schema:**
```json
{
  "user_goal": "Book a hotel in Paris for next weekend under 200 EUR",
  "persona": "business traveler, terse communication style",
  "disambiguation_points": [
    {
      "before_tool_index": 0,
      "missing_field": "budget",
      "assistant_question": "What's your budget range per night?"
    }
  ],
  "tool_sequence": ["hotels/search", "hotels/book"],
  "estimated_turns": 5,
  "conversation_type": "sequential"
}
```

**Why the Planner exists as a separate agent**: without pre-planning, the User and Assistant agents tend to produce conversations where disambiguation is either absent (assistant assumes all values) or excessive (assistant asks about things it could infer). The Planner grounds disambiguation in the actual missing required fields of the tool chain.

#### User Simulator Agent (free text)

The User agent receives:
- The user goal and persona from the ConversationPlan
- The full conversation history so far
- A signal: "it is your turn to respond"

It does **not** receive the tool chain or disambiguation plan - it only knows its goal, which makes its responses more realistic. It naturally withholds information (budget, dates) unless asked.

#### Assistant Agent (structured tool calls, free text narrative)

The Assistant receives:
- The full conversation history
- The list of available tools (the sampled chain tools, formatted as tool schemas)
- System prompt establishing its role

At each turn it chooses: clarify, call a tool, or respond. Tool calls are emitted as structured JSON per the tool schema. After receiving a tool output, the assistant generates a narrative response before deciding whether to call another tool.

**Critical design**: the assistant sees tool outputs in its conversation history, not just the tool call. This is what enables coherent chaining - the LLM "knows" what hotel_id was returned because it's in its context.

#### Judge Agent (structured output)

Receives the complete conversation transcript and scores it. See Section 6.

### 5.2 Communication Protocol

Agents communicate through a shared `ConversationContext` object - not via message passing or queues. This is a deliberate simplicity choice: for a sequential pipeline, a shared mutable object is less complexity than a message bus.

```python
@dataclass
class ConversationContext:
    plan: ConversationPlan
    tool_chain: ToolChain
    messages: List[Message]           # full conversation history
    execution_session: ExecutionSession
    metadata: Dict[str, Any]
    
    def add_message(self, role: str, content: str, tool_calls=None, tool_outputs=None):
        self.messages.append(Message(role=role, content=content, 
                                      tool_calls=tool_calls, tool_outputs=tool_outputs))
```

### 5.3 Orchestrator Loop

```
1. Sample tool chain from graph (with steering weights if enabled)
2. Planner → ConversationPlan
3. User generates opening message
4. Loop until done:
   a. Assistant generates turn
   b. If tool_call:
      - ExecutionSession.execute() → mock response
      - Append tool output to context
      - If more tools needed: continue loop (don't give back to User)
   c. If assistant asks question (detected by presence of "?" + no tool call):
      - User generates response
   d. If assistant gives final answer (no tool call, no question):
      - Mark conversation complete
5. Judge scores the completed conversation
6. If score < threshold: Repair loop
7. Write to JSONL
```

**Disambiguation detection**: whether the assistant is asking a clarifying question vs. giving a final answer is determined by a simple heuristic: does the response contain a question mark and is there no tool call? This is an acknowledged imprecision - see Section 11 for what I'd do differently.

---

## 6. Quality Evaluation Pipeline

### 6.1 Scoring Dimensions

Three dimensions were chosen to capture the aspects of conversation quality most relevant for training tool-use agents:

**Dimension 1: Tool Selection Correctness (1–5)**
Are the right tools called for the stated goal? Are required parameters populated with sensible values? Are unnecessary tool calls avoided?

*Why this dimension*: A model trained on wrong tool selections would learn to call irrelevant APIs. This is the most direct signal for "did the assistant solve the problem correctly."

**Dimension 2: Conversational Naturalness (1–5)**
Does the conversation flow naturally? Is disambiguation appropriate (not asking for information already provided, not skipping genuinely ambiguous requirements)? Does the assistant's narrative match the tool outputs?

*Why this dimension*: A dataset with robotic, formulaic conversations would train models that users find uncomfortable to interact with. Naturalness is necessary for the training data to produce human-preferred assistants.

**Dimension 3: Chaining Consistency (1–5)**
Do arguments in step N use actual values returned by step N−1? Does the final assistant response correctly reference the confirmed outputs (booking IDs, search results)?

*Why this dimension*: This is the failure mode unique to multi-step tool use and the hardest to get right. A training dataset where arguments are hallucinated would actively teach models bad chaining behavior. This dimension directly measures the correctness of the execution model.

**Composite threshold**: mean score ≥ 3.5 across all three dimensions. Any single dimension < 2.0 triggers repair regardless of mean.

### 6.2 Judge Output Schema

```json
{
  "tool_selection_score": 4,
  "tool_selection_reasoning": "Correct tools called in correct order. Budget parameter correctly passed.",
  "naturalness_score": 3,
  "naturalness_reasoning": "Disambiguation question was appropriate but phrasing was stiff.",
  "chaining_score": 5,
  "chaining_reasoning": "hotel_id from search correctly used in booking call.",
  "overall_score": 4.0,
  "repair_hints": [
    "Turn 3: assistant should ask for check-out date before calling hotels/book"
  ],
  "is_repairable": true
}
```

### 6.3 Repair Strategy

Repair is preferred over discard because generating a conversation costs multiple LLM calls - discarding wastes that compute.

**Attempt 1 - Targeted repair**: Identify the specific turns flagged in `repair_hints`. Regenerate only those turns with the hint injected into the assistant's prompt. All other turns remain unchanged.

**Attempt 2 - Full re-generation with failure context**: Keep the same tool chain but re-run the full conversation generation with the failure reasons appended to the Planner's prompt as negative examples ("in a prior attempt, the assistant failed to ask for check-out date - ensure this is explicitly requested").

**Attempt 3 - Abandon**: Log the conversation as failed (with all scores and hints). Count toward failure metrics. Do not write to output.

**When is repair not attempted?** If `is_repairable: false` from the judge (e.g., the tool chain itself is semantically nonsensical), skip directly to Attempt 2.

---

## 7. Context Management Design

### 7.1 Within-Conversation Grounding

**Approach: Full history injection + Execution Session state**

The assistant agent always receives the complete conversation history, including all tool outputs. This is the simplest correct approach: the LLM's context window becomes the "working memory" for the conversation.

The Execution Session complements this: when the assistant emits a tool call with arguments, the session's `_resolve_args` method substitutes any argument whose name matches a field in the session state with the actual returned value. This is a deterministic override - it doesn't rely on the LLM to "remember" the right ID.

**Why two mechanisms?** The full history grounds the LLM's understanding of the conversation (it knows what happened and why). The session state is a safety net that prevents hallucinated IDs even if the LLM's attention doesn't properly retrieve the right value from a long context.

**Where this breaks down:**
- Conversations > ~15 turns start approaching context limits for smaller LLMs
- Tool outputs with large result sets (e.g., search returning 100 items) consume context fast
- Mitigation: truncate tool outputs to 500 tokens; keep only the first 3–5 results from arrays

**What I'd do at scale:** A sliding window with structured extraction - summarize completed tool-call groups into a compact "state summary" rather than keeping raw outputs. At 50+ turns, a retrieval layer that fetches relevant prior outputs by field name would outperform full injection.

### 7.2 Cross-Conversation Steering

**Approach: Inverse-frequency weighted sampling with CoverageTracker**

The CoverageTracker maintains:
```python
@dataclass
class CoverageTracker:
    tool_use_counts: Counter[str]          # tool_id → times used
    tool_pair_counts: Counter[Tuple]       # (tool_a, tool_b) → times co-occurring
    domain_counts: Counter[str]            # category → times appeared
    pattern_counts: Counter[str]           # "sequential"|"parallel" → count
```

Before each graph walk, the sampler queries the tracker for node weights:
```
weight(tool) = 1 / (1 + log(1 + tool_use_counts[tool.id]))
```

This is the same inverse document frequency (IDF) weighting used in information retrieval - tools that appear frequently are downweighted, rare tools get higher probability. The log dampens the effect so a tool used 100× isn't 100× less likely than one used 1×.

**mem0 integration**: The tracker persists to disk via JSON for small runs. For larger runs or across pipeline restarts, conversations are also stored as embeddings in mem0 - this enables semantic similarity queries like "have we already generated a conversation semantically similar to this one?" before committing to generate it. This catches cases where two different tool combinations produce essentially the same user scenario.

**Non-determinism caveat**: mem0's ANN search is non-deterministic. The primary steering mechanism (counter-based weights) is fully deterministic and seeded. The mem0 semantic check is a soft signal - if it says "similar conversation exists," we add a 50% probability of re-sampling rather than forcing a resample. This way non-determinism affects distribution shape but not correctness, and results remain statistically comparable across runs.

**Where this breaks down:**
- Counter-based steering treats semantically equivalent tools (two different hotel search APIs) as distinct - it would happily over-generate hotel conversations using alternating APIs
- At 10K+ conversations, the counter becomes a bottleneck (though still O(1) lookup)
- The semantic check via mem0 partially addresses the first issue but adds non-determinism

**What I'd do at scale:** Cluster tools by embedding similarity at build time. Track coverage at the cluster level, not the individual tool level. This guarantees diversity across semantic categories, not just API names.

### 7.3 Disabling Steering (Run A vs B)

The `generate` CLI exposes `--no-cross-conversation-steering`. When set:
- CoverageTracker is initialized but its weights are not applied to sampling
- Conversations are still recorded to the tracker (for metrics comparison)
- All other pipeline behavior is identical

This allows exact A/B comparison with the same seed.

---

## 8. Prompt Design

### 8.1 Planner Agent Prompt

**Current version:**
```
You are a conversation planner for a tool-use dataset.

Given this tool chain:
{tool_chain_json}

And this tool registry context:
{tool_descriptions}

Produce a JSON conversation plan with:
- user_goal: a natural language description of what the user wants to accomplish
- persona: the user's communication style (terse/verbose/technical/casual)
- disambiguation_points: a list of {before_tool_index, missing_field, assistant_question} 
  for each required parameter that is not inferrable from the user's opening message
- estimated_turns: integer
- conversation_type: "sequential" | "parallel" | "mixed"

Rules:
- The user_goal must be achievable using exactly the tools in the chain, no more, no less
- disambiguation_points should be non-empty only for truly ambiguous required parameters
- Do not include optional parameters in disambiguation_points
- Output valid JSON only, no prose
```

**Why structured this way:**
- Tool chain is injected as JSON (not described in prose) so the LLM can see exact field names
- "no more, no less" constraint prevents the planner from inventing tool calls not in the chain
- "truly ambiguous" qualifier is important - without it, the planner adds disambiguation for every parameter, producing unrealistically interrogative assistants

**An iteration that didn't work:**
Early version omitted the `persona` field and described the user goal as a single sentence without constraints. The result: the planner generated goals that were either too broad ("I want to travel") or too narrow ("I want to book hotel ID 12345 with check-in 2026-04-11"). Neither is natural. Adding the persona field and the "achievable using exactly the tools in the chain" constraint produced much more grounded, realistic goals.

A second failed iteration asked the planner to also generate the opening user message. This made the planner output too long and the LLM frequently dropped the JSON structure in favor of including the actual conversation. Separating "plan" from "generate" fixed this.

### 8.2 Assistant Agent Prompt

**Current version:**
```
You are a helpful assistant with access to the following tools:
{tool_schemas_json}

The user wants to: {user_goal}

You have these tools available. When you need to use a tool, emit a JSON tool call 
in this exact format:
{"tool_call": {"endpoint": "<tool_id>", "arguments": {<args>}}}

Rules:
- Only call tools that are in the provided list
- Arguments must use values from the conversation - never invent IDs or references
- If a required parameter is missing, ask the user for it before calling the tool
- After receiving a tool result, incorporate it into your response naturally
- When the task is complete, give a final summary response with no tool call
```

**Why "never invent IDs"**: This is explicit because LLMs strongly tend to hallucinate plausible IDs. Making it an explicit prohibition and repeating it in the system prompt reduces (but doesn't eliminate) hallucination. The execution session's `_resolve_args` is the deterministic safety net.

**An iteration that didn't work:**
First version gave the assistant the full conversation plan including `disambiguation_points`. This caused the assistant to ask the pre-planned questions verbatim, producing stilted conversations where the phrasing was clearly robotic. Removing the plan from the assistant's context (it only sees the conversation history and available tools) produced more natural disambiguation.

### 8.3 Judge Agent Prompt

**Current version:**
```
You are evaluating a multi-step tool-use conversation for dataset quality.

Rate the conversation on three dimensions, 1–5 each:

1. Tool Selection Correctness: Were the right tools called? Were parameters sensible?
2. Conversational Naturalness: Did the conversation flow naturally? Was disambiguation appropriate?
3. Chaining Consistency: Did later tool calls use actual values from earlier outputs?

For each dimension, provide:
- A score (integer 1–5)
- One sentence of reasoning

Then provide:
- repair_hints: list of specific, actionable fixes (empty list if score >= 4.0 overall)
- is_repairable: true if targeted fixes can bring score above 3.5, false if fundamental

Respond with valid JSON only.

Conversation:
{conversation_json}
```

**Design rationale:** Judges that are given too much latitude produce inconsistent scores. The one-sentence reasoning constraint forces the judge to be specific, which produces more actionable repair hints.

---

## 9. Diversity & Quality Analysis

### 9.1 Diversity Metrics

**Metric 1: Tool-Pair Type-Token Ratio (TTR)**

```
TTR = |unique (tool_a, tool_b) pairs| / |total (tool_a, tool_b) co-occurrences|
```

Range: 0 (all conversations use the same pair) to 1 (every conversation uses a unique pair). This directly measures whether steering is preventing repetitive tool combinations.

**Metric 2: Domain Entropy**

```
H = -Σ p(domain_i) * log(p(domain_i))
```

Normalized to [0, 1] by dividing by log(|domains|). A uniform distribution across domains gives H=1; a dataset dominated by one domain gives H near 0.

**Why these two metrics:** TTR measures micro-diversity (specific tool combinations) while entropy measures macro-diversity (domain balance). Steering acts on both: it downweights overused tools (improves TTR) and overrepresented domains (improves entropy). If steering works, both metrics should improve in Run B vs. Run A.

### 9.2 Experimental Results

Runs performed at N=200 conversations each, seed=42, model=gpt-4o-mini.

| Metric | Run A (no steering) | Run B (with steering) | Change |
|--------|--------------------|-----------------------|--------|
| Tool-Pair TTR | 0.2706 | 0.3132 | +0.0426 ▲ |
| Domain Entropy (normalized) | 0.9199 | 0.9486 | +0.0287 ▲ |
| Domains seen | 10 | 10 | — |
| Unique tool pairs | 82 | 83 | +1 ▲ |
| Mean Judge Score (overall) | 4.38 | 4.37 | -0.01 ▼ |
| Mean Tool Selection Score | 4.41 | 4.40 | -0.01 ▼ |
| Mean Naturalness Score | 4.57 | 4.54 | -0.03 ▼ |
| Mean Chaining Score | 4.17 | 4.17 | 0.00 — |
| Pass rate (≥3.5 overall) | 100.0% | 100.0% | — |
| Mean turns per conversation | 16.64 | 16.05 | -0.59 |
| Mean tool calls per conversation | 3.06 | 2.94 | -0.12 |

### 9.3 Diversity–Quality Tradeoff Analysis

**Results summary:** Steering improved both diversity metrics — Tool-Pair TTR (+0.0426: 0.2706 → 0.3132) and domain entropy (+0.0287: 0.9199 → 0.9486) — while quality was essentially unchanged (-0.01 overall, well within noise). The pair-level cooldown (implemented in `CoverageTracker`, see §12 and the concrete fix in the earlier iteration) resolved the TTR regression observed in prior runs at smaller N.

**Why the earlier TTR regression happened.** Prior runs (N=10, N=30, N=100 without the cooldown) showed TTR moving in the wrong direction. Root cause: IDF-based node weights push the sampler toward underused tools, but in a graph with only 31 nodes and 81 edges (avg 2.6 edges/node), underused tools have fewer outgoing edges. The random walk funnels through a small set of highly-connected bridge nodes regardless of starting point, concentrating pair usage. The pair-level cooldown addresses this directly — by setting edge weight to 0 for recently-used pairs, it forces the walk to find alternative continuations rather than routing through the same bridges.

**Why the fix works.** The cooldown and IDF weights are orthogonal mechanisms: IDF handles domain balance (which metric benefited in prior runs too), the cooldown handles pair-level repetition. At N=200, both mechanisms together deliver improvement on both metrics simultaneously.

**The quality cost is negligible.** The -0.01 overall score difference is well within one standard deviation of both distributions and is not statistically significant at N=200. Chaining score is identical (4.17 vs 4.17). Steering does not degrade quality — the diversity–quality tradeoff hypothesis is not supported by this data. Both can improve together.

**Where this still breaks down at scale.** The cooldown window (K=10) is corpus-size-sensitive: at 10K+ conversations, a window of 10 pairs becomes negligible relative to the combinatorial space. At that scale, steering at the cluster level (embed tools, cluster by semantic similarity, track coverage at the cluster level) is the right approach — it guarantees semantic diversity rather than just name-diversity. See §11 for the full scale-out plan.

---

## 10. Output Schema

Each JSONL record:

```json
{
  "conversation_id": "conv_0042",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [
      {"endpoint": "hotels/search", "arguments": {"city": "Paris", "max_price": 200}}
    ]},
    {"role": "tool", "tool_id": "hotels/search", "content": {"results": [...]}},
    {"role": "assistant", "content": "I found a hotel for you..."}
  ],
  "judge_scores": {
    "tool_selection": 4,
    "naturalness": 3,
    "chaining": 5,
    "overall": 4.0,
    "reasoning": {...}
  },
  "metadata": {
    "seed": 42,
    "tools_used": ["hotels/search", "hotels/book"],
    "tool_categories": ["Travel"],
    "num_turns": 7,
    "num_tool_calls": 2,
    "num_distinct_tools": 2,
    "conversation_type": "sequential",
    "disambiguation_turns": 1,
    "repair_attempts": 0,
    "steering_enabled": true,
    "generated_at": "2026-04-09T12:00:00Z",
    "model": "gpt-4o-mini"
  }
}
```

**Schema decisions:**
- `messages` follows the OpenAI message format: role-tagged, tool_calls as a list on assistant messages, tool role for outputs. This makes the dataset directly usable for fine-tuning on OpenAI-compatible APIs.
- `judge_scores.reasoning` is preserved (not just scores) so researchers can filter by specific failure modes.
- `metadata.repair_attempts` enables analysis of dataset quality by generation difficulty.
- `metadata.steering_enabled` enables post-hoc filtering to compare Run A and Run B from the same file.

---

## 11. What I Would Do Differently at Scale

**Tool Graph**: Replace name-matching edge detection with a two-stage approach: (1) embed all parameter and response field descriptions, (2) run approximate nearest-neighbor search to find compatible pairs. This catches semantic matches that name matching misses (e.g., `hotel_result` → `accommodation_id`). Cost: O(n) embedding calls at build time, but graph building is a one-time operation.

**Context management**: Replace full history injection with a structured state extractor that maintains a compact "conversation state dict" - key entities (IDs, names, dates) extracted from all prior tool outputs. Inject this dict rather than full raw outputs. This scales to 50+ turn conversations without hitting context limits.

**Mock generation**: Invest in an LLM-generated mock cache - generate high-quality mocks for each tool once, store them, reuse with parameterized substitution. This gives LLM-quality mocks at schema-derived mock cost.

**Disambiguation detection**: Replace the question-mark heuristic with a small classifier (even a prompted LLM with cached results) that distinguishes "clarifying question", "tool-calling intent", and "final answer". The heuristic fails on rhetorical questions and multi-sentence turns that contain both a question and a tool call intent.

**Scale-out**: The current sequential per-conversation pipeline is bottlenecked by LLM latency. At 10K+ conversations, use an async pipeline with a work queue (asyncio + producer/consumer pattern), batching LLM calls where possible, and parallelizing conversation generation across a thread pool.

---

## 12. Observed Failure Modes & Fixes

This section documents failure modes observed during development, their diagnosed root causes, and the fixes applied or proposed. It is included because understanding where a system breaks - and why - is as important as knowing where it works.

---

### 12.1 User Agent Role Confusion

**Observed failure modes:**
- User says "I don't have access to real-time data" - breaking the human persona entirely
- User echoes the assistant's last message back verbatim (copy-paste echo bug)
- User offers to help the assistant ("Let me check for you", "How about I help you find…") - adopting the assistant's role
- User speaks in third person about themselves ("you're flying from LAX to NYC")

**Root cause - prompt framing:** The original prompt used "ROLEPLAYING" and "NOT an AI" - phrases that prime the LLM to think about its AI identity rather than suppress it. The more effective frame is to never mention AI at all and assert the human identity directly and briefly.

**Root cause - history bleed:** The `respond()` method was passing full assistant messages (verbose, tool-aware, AI-flavoured) to the user agent. The user agent absorbed this framing and started mirroring the assistant's language patterns. Fix: truncate assistant messages to ~120 chars in the user agent's history view so it knows what happened but doesn't absorb the style.

**Root cause - echo bug:** When the orchestrator calls `user_agent.respond()` after the assistant gives a closing message, the user agent sees the closing line as the last message in context and mirrors it. Fix: (a) add explicit "NEVER repeat what the assistant just said" rule, (b) pass a `hint` parameter for extension turns so the user agent has a specific topic to raise rather than generating from a closing context.

**Root cause - steering hint injection:** The orchestrator injected system-level steering messages ("Good. You already have the search results. Please now proceed with the next step of the task.") as user messages. These appear verbatim in the output JSONL as if a human said them - completely breaking immersion. Fix: rephrase steering hints as natural human utterances ("Okay, what's the next step from here?").

**Proposed fix at scale:** A dedicated user persona module that maintains a persistent persona state (name, communication style, known facts about their goal) and generates responses from that state rather than from raw conversation history. This isolates the user agent from assistant-framing contamination entirely.

---

### 12.2 Mock Data Coherence Issues

**Observed failure modes:**
- Business names generated as human names: "Joe Martinez", "Brian Burton" (hotel, restaurant names)
- `status` fields: random words like "threat", "road", "surface" (from `faker.word()`)
- `added_at`, `created_at`, `timestamp` fields: random words instead of datetimes ("easy", "road")
- `stops` field: float value 1.62 (should be integer 0–3)
- `available_rooms`: float 17.49, 30.58 (should be integer)
- `comment`/`review` fields: single random words instead of sentences
- `reviewer_name` generating company names instead of person names
- Stock `symbol` field returning a random ticker (AMZN) when called with a specific symbol (MSFT)
- Current stock `price` and historical `open`/`close` generated independently with no relationship - leads to $22 current price but $574 historical open in the same conversation

**Root cause:** Field values are generated independently without cross-field awareness or input-arg echoing. The `_FIELD_HINTS` dict used `faker.word()` for `status` (wrong) and had no entries for `comment`, `review`, datetime fields, or integer-count fields.

**Fixes applied:**
- `name` field: context-sensitive - `faker.company()` for business categories, `faker.name()` for person contexts (reviewer, author, customer)
- `status`: fixed vocabulary (`confirmed`, `pending`, `active`, `success`, `completed`)
- Timestamp fields (`_at`, `created`, `updated`, `booked`): recent ISO datetime
- Integer count fields (`rooms`, `spots`, `stops`, `guests`): cast to `int`
- `comment`, `review`, `feedback`: `faker.sentence()`
- Input-arg echoing: if response field name matches an input arg (e.g. `symbol`), echo the input value

**Remaining known issue - cross-call price incoherence:** When a conversation calls both `get_stock_quote` and `get_stock_history`, the current price and historical prices are generated by separate mock calls with no shared state. A stock showing $22 current price but $574 historical open is mathematically impossible. Fix at scale: a mock cache keyed by `(tool_category, symbol/entity)` that ensures the same entity returns consistent numeric ranges across all calls in a conversation.

---

### 12.3 Conversation Quality Gaps

**Observed:** Assistant asks 6 questions at once in a numbered list rather than disambiguating one question at a time.
**Root cause:** Assistant system prompt didn't constrain question quantity.
**Fix applied:** Added explicit rule: "Ask for ONE missing piece of information at a time, not a numbered list."

**Observed:** Conversations run very long (23–37 turns) for simple 2–3 tool tasks.
**Root cause:** The assistant loops on disambiguation, and the orchestrator's max-turn limit (20) is occasionally exceeded because both user extensions and repair turns add to turn count.
**Proposed fix:** Tighten the max-turn limit relative to chain length: `max_turns = max(10, len(chain_tools) * 6)` instead of a flat 20.

**Observed:** `check_in_date` / `check_out_date` passed as `2023-11-08` when the conversation context implies a future date. The planner generates dates in the past.
**Root cause:** The planner generates date strings without awareness of the current date.
**Proposed fix:** Inject `current_date` into the planner prompt so generated dates are always in the future.
