*This project has been created as part of the 42 curriculum by lbalderr.*

# Call me maybe

## Description

**call me maybe** is a constraint-based LLM generation engine designed to enforce strict, type-safe JSON function calls. By utilizing state machines, logit masking, and token prefix trees, this engine forces a language model to output structurally perfect JSON payloads that map precisely to predefined function schemas.

The goal of the project is to bridge the gap between unstructured probabilistic text generation and the rigid structural requirements of traditional backend systems, guaranteeing parsable JSON without relying on prompt engineering alone.

## Instructions

### Prerequisites

- Python 3.10+
- `uv` package manager

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lucasbpaiva/call_me_maybe.git
   cd call_me_maybe
   ```

2. Initialize the environment and install dependencies:

   ```bash
   uv sync
   ```

### Execution

Run the generation engine using the provided Makefile:

```bash
make run
```

Or manually via the CLI:

```bash
uv run python -m src --functions_definition <function_definition_file> --input <input_file> --output <output_file>
```

## Algorithm Explanation

The core engine relies on **Constrained Decoding via Logit Masking**. Instead of allowing the LLM to sample freely from its vocabulary (~150,000 tokens), the engine intercepts the score distribution (logits) at every token generation step.

1. **State Machines and Predicates:** For unbounded types (numbers, strings), the engine uses regular expressions (`NumberValidator`, `StringValidator`) to determine whether a token legally continues a JSON primitive or properly escapes a string.
2. **Prefix Trees (Tries):** For bounded types (function names, booleans), a trie evaluates whether a token builds toward a valid path (e.g., `"true"` or `"false"`).
3. **Lazy Evaluation:** The `pick_best_allowed` algorithm sorts the model's logits in descending order and lazily evaluates candidates against the current state machine. The first token to return `True` from the predicate is forced into the input sequence.
4. **Token Normalization:** Because BPE (Byte Pair Encoding) tokens can overlap or behave unexpectedly at word boundaries, generated primitives (like numbers and booleans) are temporarily held in a string buffer, fully parsed, and then cleanly re-tokenized before being appended to the root context.

## Design Decisions

- **Forced JSON Initialization:** To prevent trailing garbage and guarantee structural integrity, the orchestrator manually injects `{"name": "` into the context before generating the function name, guaranteeing the LLM starts inside a valid JSON object.
- **Schema Decoupling:** The `returns` type is defined in the Pydantic schema for backend typing and evaluation, but is stripped via `exclude={"returns"}` during prompt construction to save context window tokens and prevent LLM confusion.
- **Fault Isolation:** The main orchestrator wraps generation in a `try...except` loop. If an edge case crashes the generator, a dictionary is injected with a `None` name and an error trace, ensuring downstream parsing never encounters an unexpected `KeyError` and batch processing is never halted.
- **Type Safety:** Strict `mypy` typing is enforced throughout, including the use of Pydantic models for I/O validation.

## Performance Analysis

### Speed

**Where the time goes.** Profiling with `cProfile` showed that runtime is dominated by the model, not by the constraint logic. In the initial version, `get_logits_from_input_ids` accounted for about 78% of a 278 s run (130 forward passes at about 1.67 s each, with a prompt of about 465 tokens). The Python-side masking cost about 20 s in total. The SDK exposes no public KV cache, so every forward pass reprocesses the entire prompt, which makes the cost per step scale roughly linearly with prompt length.

**Optimizations applied.**

| Optimization | Effect |
|---|---|
| Lazy masking (`pick_best_allowed`): sort logits once, test candidates in descending order against a per-token predicate | Replaces a ~150k-token scan per step with 1~5 candidate checks on average |
| Compact prompt: Dropped the JSON indentation, effectively removing spaces, tabs and newlines the model doesn't need. Also omitted the `returns` parameter | Prompt reduced from ~465 to ~238 tokens, forward pass from ~1.67 s to 0.95 s |


**Results:**

| Version | Total time (11 prompts, school machine) | Forward passes | Time per pass |
|---|---|---|---|
| Baseline | 278 s | 130 | 1.67 s |
| Final | 135 s | 137 | 0.95 s |

**Limits.** Speed still depends on the hardware (CPU `float32` at 42 clusters, MPS on Apple Silicon), on the number and verbosity of function definitions (they extend the prompt), and on how many tokens a string or number takes. Measured time per prompt is 12.31 s on average. Extrapolating linearly, the 5-minute budget supports roughly 24 prompts of this complexity.

### Accuracy

Two different properties are measured separately.

**Structural validity: 100% by construction.** The JSON skeleton (keys, braces, separators) is injected by the orchestrator rather than generated. Values are produced only from tokens that the number, string, boolean, or trie predicates accept, so the model has no way to emit a malformed value. Every output was parsed with `json.loads` and validated against the Pydantic schema during evaluation, with 0 failures over 42 prompts.

**Semantic correctness: measured, not guaranteed.** Constraining the format cannot make the model choose the right function or extract the right argument. On the custom 42-prompt evaluation set (`eval_cases.json`):

| Metric | Result |
|---|---|
| Routing accuracy (correct function) | 92.9% |
| Parameter accuracy (correct arguments, types included) | 76.2% |
| Strict accuracy (both correct, no failure) | 76.2% |

Failure modes observed incclude: dropped negative signs on numbers, couldn't generate a regex pattern for extracting emails from a string. These come from the 0.6B model's limited reasoning, not from the decoder, so the constraint layer cannot fix them.

### Reliability

- **Fault isolation:** an exception on one prompt is caught, logged with the offending prompt, and recorded in the result (`name = None` plus an error trace), so the remaining prompts still run.
- **Input robustness:** missing files, empty files, invalid JSON, wrong shapes, and an empty function list produce a clear error message instead of a traceback.
- **Resource handling:** files are opened with context managers, and the output directory is created if missing.
- **Known limits:** only string, number, and boolean parameter types are supported.

## Challenges Faced

1. **Storage limitation:** The machines at 42 school have limited storage, so I had to implement a Makefile rule to redirect all huggingface and model caches to another directory with enough storage.
2. **Ensuring 100% valid JSON generation:** While generating forced parts of the JSON structure manually and letting the model pick only the function name and parameters improved performance, I had a lot of bugs where the model would also want to output part of the JSON structure, leading to invalid syntax like extra quotes or braces.
3. **Improving performance:** Even though the program was returning accurate results, it was running too slow on the 42 school machines, so I had to look for way of optimizing it. In the end I had to shrink the prompt size to reduce the number of tokens for each prompt and improve the token masking step to reduce the time for each forward pass.

## Testing Strategy

The implementation is validated via a custom evaluation script (`src/evaluate.py`) against a rigorous ground-truth test suite (`eval_cases.json`).

- **Test Matrix:** 40+ prompts covering straightforward instructions, ambiguous requests, semantic distractors, and severe edge cases (extremely large numbers like `-1.79e308`, empty strings, unescaped quotes, and complex regex formatting).
- **Metrics Tracked:**
  - *Routing Accuracy:* Did the model pick the correct function schema?
  - *Parameter Accuracy:* Did the state machine perfectly extract and type the parameters?
  - *Strict Accuracy:* Did both succeed without a crash?
- **Execution:** Run `make eval` to measure wall-clock generation time and immediately print the accuracy delta.

## Example Usage

**Input (`prompts.json`):**

```json
[
  {
    "prompt": "Set the cellar to 55 degrees with eco mode enabled"
  }
]
```

**Functions (`functions.json`):**

```json
[
  {
    "name": "fn_set_thermostat",
    "description": "Sets the room temperature.",
    "parameters": {
      "room": { "type": "string" },
      "temperature": { "type": "number" },
      "eco_mode": { "type": "boolean" }
    }
  }
]
```

**Output:**

```json
[
  {
    "prompt": "Set the cellar to 55 degrees with eco mode enabled",
    "name": "fn_set_thermostat",
    "parameters": {
      "room": "cellar",
      "temperature": 55,
      "eco_mode": true
    }
  }
]
```

## Resources

- **Regex Documentation:** [Python `re` module](https://docs.python.org/3/library/re.html), used to structure the JSON string and number state machines.
- **Trie Data Structures:** General algorithmic concepts for prefix trees.
- **AI Usage:** Google Gemini was used as an interactive thought partner and pair-programming assistant during development. Specifically, it was used to:
  - Brainstorm and architect the `pick_best_allowed` lazy evaluation optimization.
  - Refine the `StringValidator` regex to handle strict JSON escape sequences.
  - Generate the edge-case evaluation suite (`eval_cases.json` and `eval_functions.json`).
  - Scaffold Python Google-style docstrings and identify the root cause of the `Extra data` JSON parsing errors.
