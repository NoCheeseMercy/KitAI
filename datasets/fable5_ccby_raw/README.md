---
license: cc-by-4.0
pretty_name: Fable 5 Claude Code Traces
task_categories:
- text-generation
language:
- en
tags:
- agent-traces
- format:agent-traces
- traces
- coding-agent
- agentic-coding
- tool-use
- claude-code
- fable-5
- claude-fable-5
- teich
- software-engineering
- reasoning-traces
- open-data
- distillation
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: "*.jsonl"
---

# Fable 5 Claude Code Traces

A full, scrubbed release of Fable 5 Claude Code session traces for researchers studying real coding-agent behavior: multi-turn prompts, assistant responses, tool calls, command output, retries, and session-level workflow metadata.

This release keeps the full package intact: **18 sessions**, **9,497 JSONL events**, **0 excluded rows**, and **0 quarantine files**. The traces are preserved in the native agent-event format so they can be inspected in Hugging Face Agent Traces / Data Studio and parsed directly for custom analysis.

## Why this dataset may be useful

- Study how coding agents interleave natural-language reasoning, shell/file tools, observations, retries, and task management.
- Build or evaluate tool-use policies, trace parsers, distillation pipelines, and agent-observability tooling.
- Inspect complete session timelines rather than isolated prompt/completion pairs.
- Compare raw agent-event logs with converted training formats used by other Fable 5 trace releases.

## Dataset contents

| Item | Count |
|---|---:|
| Claude Code session files | 18 |
| JSONL events | 9,497 |
| Assistant events | 2,506 |
| User events | 1,424 |
| Attachment / tool-result events | 3,428 |
| Excluded files | 0 |
| Excluded rows | 0 |

Files at the repository root are one scrubbed session per `*.jsonl` file. Each line is one event. `manifest.json` contains per-file row counts, event counts, scrub-count summaries, and totals.

Common event types include `assistant`, `user`, `system`, `attachment`, `last-prompt`, `mode`, `permission-mode`, `ai-title`, `file-history-snapshot`, `queue-operation`, and `agent-name`. Common keys include `type`, `sessionId`, `uuid`, `parentUuid`, `timestamp`, `message`, `command`, `toolUseID`, `exitCode`, and `durationMs`.

## Loading

### Session-level view with `datasets`

Hugging Face recognizes these files as agent traces. `load_dataset` returns one row per session, with the raw event stream nested in `trace`:

```python
from datasets import load_dataset

REVISION = "v1.0-full-scrubbed"  # pinned full data release

ds = load_dataset(
    "AlinCiocan/fable-5-claude-code-traces",
    split="train",
    revision=REVISION,
)

print(ds)                 # 18 session rows
print(ds[0].keys())
print(len(ds[0]["trace"])) # events in the first session
```

Typical columns are `harness`, `session_id`, `prompt`, `messages`, `tools`, `metadata`, `sent_at`, `num_user_messages`, `num_tool_calls`, `trace`, and `file_path`. The data files were first uploaded at commit `620a9ad328563e66a041b3eb21f59c48361ed4d4`; the `v1.0-full-scrubbed` tag pins the public-ready release.

### Raw event iteration

Use the raw files when you need all 9,497 events as individual JSONL records:

```python
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REVISION = "v1.0-full-scrubbed"
root = Path(snapshot_download(
    repo_id="AlinCiocan/fable-5-claude-code-traces",
    repo_type="dataset",
    revision=REVISION,
    allow_patterns=["*.jsonl", "manifest.json"],
))

for path in sorted(root.glob("*.jsonl")):
    with path.open(encoding="utf-8") as f:
        for line in f:
            event = json.loads(line)
            # analyze event
```

## Scrubbing and privacy

The source traces were scrubbed before publication with a deterministic recursive scrubber over JSON keys and values. Private identifiers were replaced with stable typed placeholders while preserving the research-relevant structure of the sessions.

Preserved where possible: roles, timestamps, model/tool labels, tool-call structure, command shape, exit codes, debug flow, and public technical labels.

Replaced categories include personal/account identifiers, host/device names, private projects and domains, local paths, LAN/private IPs, phone numbers, Telegram IDs, tokens and secret assignments, config dumps, private media filenames, and private event terms.

Anonymization is best-effort and should not be treated as a mathematical guarantee. These are real coding-agent traces and may contain generated code, command output, tool observations, and references to local tooling. If you find sensitive material that should be removed, please open a discussion on the dataset Community tab.

## Intended use

This dataset is intended for research and tooling around coding agents, including trace analysis, qualitative workflow study, tool-call modeling, data conversion, redaction evaluation, and agent observability.

It is not a benchmark, not a live tool-execution environment, and not a representative sample of all Fable 5 or Claude Code usage. Tool outputs and commands are historical logs and are not guaranteed to run elsewhere.

## License and attribution

The dataset package is released under **CC-BY-4.0**. Please attribute this dataset when using or redistributing it.

Trace contents may include code snippets, logs, model outputs, and references to third-party tools or projects; downstream users are responsible for checking that their use complies with any applicable upstream terms or licenses.

This dataset is not affiliated with or endorsed by Anthropic, Hugging Face, TeichAI, or any other agent/model vendor. Product and project names are used only to identify the systems that produced or consume the traces.

## Related resources

- Hugging Face Agent Traces documentation: https://huggingface.co/docs/hub/en/agent-traces
- Trace Commons agent-trace dataset: https://huggingface.co/datasets/trace-commons/agent-traces
- Teich trace tooling: https://github.com/TeichAI/teich
- Fable 5 trace pool: https://huggingface.co/datasets/armand0e/claude-fable-5-claude-code
- Processed Fable 5 trace release: https://huggingface.co/datasets/Glint-Research/Fable-5-traces
- Related Fable 5 trace releases may overlap in source material; deduplicate before mixing them for training or evaluation.

## Citation

```bibtex
@misc{ciocan2026fable5claudecodetraces,
  title = {Fable 5 Claude Code Traces},
  author = {Ciocan, Alin},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/AlinCiocan/fable-5-claude-code-traces}},
  license = {CC-BY-4.0}
}
```
