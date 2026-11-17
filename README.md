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

The implementation was validated via a custom evaluation script comparing the produced output from (`eval_cases.json`) against the expected results written manually.

- **Test Matrix:** 40+ prompts covering straightforward instructions, ambiguous requests, semantic distractors.
- **Metrics Tracked:**
  - *Routing Accuracy:* Did the model pick the correct function schema?
  - *Parameter Accuracy:* Did the state machine perfectly extract the type and parameters?
  - *Strict Accuracy:* Did both succeed without a crash?

## Example Usage

To use this program you will need a JSON with an array of objects containing the prompts, like this one:

**Input (`prompts.json`):**

```json
[
  {
    "prompt": "Set the cellar to 55 degrees with eco mode enabled"
  }
]
```

You will also need another JSON file with an array of function definitions, they need to respect the schema: `name`, `description`, `parameters`, where `parameters` is an object containing each parameter of the function with an object containing their `type` as the value.

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

Then run the program via CLI with uv:

```bash
uv run python -m src --functions_definition functins.json --input prompts.json --output results.json
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

## Bonus Features

In addition to core constrained decoding, this project implements three architectural bonuses focused on reliability, self-contained tokenization, and automated verification:

### 1. Custom Byte-Level BPE Tokenizer (`encode` & `decode`)

Rather than relying on `Small_LLM_Model`'s built-in `encode()` and `decode()` methods, the pipeline uses a custom, standalone Byte-Level BPE tokenizer (`src/tokenizer.py`) initialized directly from `tokenizer.json`:

* **Public `encode(text: str) -> list[int]` Pipeline:** Partitions special and added tokens longest-first prior to text normalization (`SpecialTokens`), applies Unicode `NFC` normalization to ordinary text segments, isolates words/digits/punctuation using Qwen's Unicode-aware regular expression (`PreTokenizer`), maps raw UTF-8 bytes onto a 256-character printable Unicode alphabet (`ByteMap`), and executes deterministic rank-ordered BPE merges with per-word caching (`BPE`).


* **Public `decode(ids: list[int], skip_special_tokens: bool = True) -> str` Pipeline:** Buffers consecutive BPE symbols before reversing the `ByteMap` translation and performing a single UTF-8 decode per contiguous byte run. This preserves multi-byte characters (such as CJK text and astral-plane emojis) split across multiple tokens while injecting literal added tokens cleanly.


* **Full Pipeline Integration:** Powers prompt and delimiter tokenization, vocabulary table construction (`build_token_dictionary`), and final JSON output reconstruction (`generate_function_call`) with 100% token-for-token parity against the SDK.



### 2. Error Recovery & Retry Mechanism

To prevent malformed outputs or dead-end token paths during constrained generation, the pipeline incorporates multi-layered validation and recovery:

* **Schema & Input Validation:** Strict Pydantic v2 models validate function definitions, input prompts, and tokenizer tables at startup, failing fast with descriptive diagnostics on corrupted configurations.


* **Finite-State Constrained Backtracking & Normalization:** State machines (`PrefixTree`, `NumberValidator`, `StringValidator`) prune invalid token candidates at every decoding step. When numeric or string segments terminate, intermediate token sequences are rolled back and re-tokenized in canonical form (e.g., normalizing parsed numbers via `float` conversion and clean escape formatting) so subsequent logits remain well-conditioned.


* **Graceful Generation Retry Loop:** If a generation attempt hits a dead-end state (such as exhausting valid vocabulary candidates or exceeding step limits without reaching a valid closing delimiter), the generator catches the failure, resets/adjusts the decoding state, and retries generation to guarantee a syntactically valid JSON payload.



### 3. Comprehensive Test & Static Analysis Suite

The codebase includes an automated testing and quality-assurance suite covering individual components, end-to-end differential parity, and strict static typing:

* **Frozen Oracle & Differential Testing (`tests/golden.json`):** Validates `encode` and `decode` byte-for-byte against `Small_LLM_Model` across a corpus of edge cases—including composed/decomposed Unicode (`NFC`/`NFD`), CJK, ZWJ emoji sequences, partial multi-byte prefix decodes, control tokens (`<|im_start|>`, `<think>`), and complete function-call JSON payloads.


* **Component Unit & Fuzz Tests:** Dedicated `pytest` modules verify `ByteMap` bijectivity, `BPE` merge-rank priority ordering, `PreTokenizer` lossless partitioning, `SpecialTokens` longest-first matching, constraint state machines, and seeded random-Unicode fuzzing.


* **Automated Execution & Linting:** Run the entire suite via `make test` and enforce `flake8` compliance alongside `mypy --strict` type safety via `make lint-strict`.

## Resources

- **Regex Documentation:** [Python `re` module](https://docs.python.org/3/library/re.html), used to structure the JSON string and number state machines.
- **GeeksforGeeks:** [Trie Data Structures](https://www.geeksforgeeks.org/dsa/trie-insert-and-search/). General algorithmic concepts for prefix trees.
- **Constrained Decoding:** [Language AI Handbook](https://mbrenndoerfer.com/writing/constrained-decoding-structured-llm-output) by Michael Brenndoerfer has a really in-depth explanation of the concept with lots of examples.
- **HuggingFace LLM Course:** [Chapter 6](https://huggingface.co/learn/llm-course/en/chapter6/1) introduces the concepts of Tokenizers, Normalization, Pre-Tokenization, Byte-Pair Encoding, and shows how to build a tokenizer from scratch.
- **MDN:** Quick [reference](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/String/normalize) for understanding NFC Normalization.

## AI Usage: 

Google Gemini and Claude were used as interactive thought partners and pair-programming assistants during development. Tasks assisted by AI include:
  - Architecturing optimizations like the `pick_best_allowed` lazy evaluation.
  - Designing and implementing the regex based validation classes like `StringValidator` and `NumberValidator`.
  - Generate prompts and functions for the test suite.
  - Generating Python Google-style docstrings and helping in writing this README.
