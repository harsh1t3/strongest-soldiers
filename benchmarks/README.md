# Benchmarks

A reproducible comparison between this fork and upstream [microsoft/tinytroupe](https://github.com/microsoft/tinytroupe).

## What it measures

`scenarios.py` holds small, realistic uses of the library's public API: running a social network, saving and
reloading an agent with its memory, creating a calendar event, randomizing an A/B test, scoring two populations
against each other, and so on. Each scenario either works or fails for a stated reason.

Every scenario is **deterministic and LLM-free**: no API key, no network access, no model quota. The same file
runs against any checkout, so the result is a like-for-like comparison rather than a claim about model quality.

## What it does not measure

Nothing here says anything about how *realistic* the simulated people are. That depends on the model behind the
library, and the two checkouts can run different models. This benchmark answers a narrower question: given the
same documented API call, does the library do what its documentation says?

## Running it

```bash
git clone https://github.com/microsoft/tinytroupe ../tinytroupe-upstream
python benchmarks/compare_with_upstream.py --baseline ../tinytroupe-upstream
```

The driver imports each checkout from its own directory, with its own `config.ini`, in a separate process, and
prints a per-scenario table. Add `--json results.json` to keep the full output, including tracebacks.

`render_comparison.py` runs the same scenarios and writes the summary image used in the main README:

```bash
python benchmarks/render_comparison.py --baseline ../tinytroupe-upstream
```

## Result (2026-10-08, upstream 0.7.0 at `a6244b3`)

**1 of 23 scenarios work upstream; 23 of 23 work here.**

| What breaks upstream | How it fails there |
|---|---|
| A social network runs its simulation steps | `TypeError: _step() got an unexpected keyword argument 'timedelta_per_step'` |
| Agents added through a relation are reachable by name | the agent is never registered, so messages to it are broadcast or dropped |
| Interventions are not shared between worlds | a second world holds the first world's intervention |
| Time can be skipped without an explicit timedelta | `TypeError: unsupported operand type(s) for *: 'int' and 'NoneType'` |
| Agents acting in parallel share the step's transaction | 7 top-level transactions instead of 1, so the cache cannot replay |
| A failing tool does not abort the agent's turn | the tool's exception ends the whole simulation |
| An agent's memory survives a save/load round trip | all 3 memory items come back as `JsonSerializableRegistry` objects |
| `auto_rename_agent` lets a specification be loaded twice | `ValueError: Agent name Agent is already in use.` |
| A shared persona fragment is not mutated | the other agent's interests become `['chess', 'surfing']` |
| `retrieve_last(0)` returns no memories | returns every memory instead |
| Repetition prevention can be turned off | the setting is ignored |
| Relationships can be declared on a fresh agent | `KeyError: 'relationships'` |
| An agent can create a calendar event | `ValueError: Invalid key date in dictionary` |
| Documents needing sanitization can be indexed | `AttributeError: 'Document' object has no attribute 'metadata_seperator'` |
| A/B randomization varies between items | all 200 items get the same order |
| Clearly different populations are not scored as equivalent | scores 1.00 similarity (1.0 means identical) for opposite data |
| A confidence interval has the sign of its difference | the difference is +10.0 but its CI is (-12.0, -7.9) |
| Statistical assumptions can be checked | `ValueError: Metric 'score' not found in control data` |
| Results survive a model that writes one object per item | the whole extraction comes back empty (`{}`) |
| A failed comparison is not reported as a real score | a made-up 0.500 is reported as a measurement and averaged in |
| A failed enrichment does not empty a document | the agent's draft is replaced by an empty document |
| A model choice containing punctuation is matched | `"C++"` is matched as `"C"` |

One scenario behaves the same in both checkouts (JSON with accented characters parses correctly in both), and no
scenario works upstream but fails here.

Beyond these scenarios, upstream needs an OpenAI or Azure API key before any simulation can run at all, while
this fork defaults to the local Claude Code CLI login (see the README's *Claude Code Support* section).
