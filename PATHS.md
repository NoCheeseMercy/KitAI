# Hardcoded Paths — Make KitAI Work on Your Device

## Why it won't run as-is

KitAI was built on one specific Windows machine, and the helper scripts, recovery launchers, and status documents in this repository hardcode the absolute paths from that machine. On your device those folders do not exist, so those files will fail until you replace them with the real paths on your device.

You have two ways to fix it:

1. **Replace every folder path by hand** (works, but tedious — ~30 files), or
2. **Let a free agent do it for you** — the easiest way. [OpenCode](https://opencode.ai) is a free, open-source terminal coding agent that can make all the changes in one go. A ready-made prompt is included below.

## The three paths you will meet

| Hardcoded value | What it is on the original machine |
|---|---|
| `D:\EYAD\KitAI` | The project root (also written `D:/EYAD/KitAI` or `D:\\EYAD\\KitAI`) |
| `C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe` | A Python 3.14 interpreter with PyTorch + CUDA installed |
| `C:\Users\abdel\AppData\Local\Programs\Ollama\ollama.exe` | The Ollama executable |

**Important:** what needs to change is not just the username folder (`abdel` → your Windows username) but the whole drive and folder layout (`D:\EYAD\KitAI` → wherever *you* cloned this repo). That is why a simple username substitution is not always enough — the paths must be replaced with the real ones on your device.

## What is affected — and what is not

**Works out of the box (no hardcoded paths):** the core framework — `models/`, `training/`, `tokenizer/`, `datasets/`, `utils/`, `tests/`, and the main entry points (`scripts/train.py`, `scripts/chat.py`, `scripts/generate.py`, etc.) when invoked with relative paths or CLI arguments.

**Needs path fixes before it will run:**

Status / documentation:
- `ACTIVE_MIXED_TRAINING_STATUS.md`
- `datasets/fable5_ccby_raw/inspection_report.json`

Repo-root helper scripts:
- `_build_overnight_config.py`
- `_inspect_checkpoint.py`
- `_inspect_resume_state.py`
- `_patch_llamacpp_kitai_tokenizer.py`
- `_patch_train_for_token_cache.py`
- `_patch_trainer_fused_option.py`
- `_patch_true_resume.py`

Launchers, watchers, exporters (in `scripts/`):
- `scripts/launch_kitai_200m_protected_rerun.ps1`
- `scripts/start_kitai_protected_on_login.ps1`
- `scripts/resume_kitai_200m_protected_from_step_00002000.ps1`
- `scripts/resume_kitai_200m_protected_from_step_00002000_attempt_02.ps1`
- `scripts/resume_kitai_200m_protected_from_step_00002000_attempt_03.ps1`
- `scripts/run_kitai_200m_final_sft.ps1`
- `scripts/export_live_protected_checkpoint_to_ollama.ps1`
- `scripts/export_stopped_protected_checkpoint.ps1`
- `scripts/export_test_protected_checkpoint.ps1`
- `scripts/watch_kitai_training.py`
- `scripts/watch_kitai_checkpoints.py`
- `scripts/finalize_kitai_run.py`
- `scripts/finalize_fable5_wikitext_run.py`
- `scripts/finalize_oasst1_wikitext_run.py`
- `scripts/curate_fable5_mix.py`
- `scripts/curate_oasst1_wikitext_mix.py`
- `scripts/inspect_fable_traces.py`
- `scripts/profile_fable_trace_content.py`

Chat launchers (repo root):
- `talk_to_kitai.bat`
- `test_kitai.bat`

## Option 1 — Replace every path by hand

1. Clone the repo and note your project folder, for example `C:\Users\<you>\Projects\KitAI`.
2. Find your Python interpreter that has PyTorch installed:

   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   where python
   ```

3. Find your Ollama executable. The default install location is `C:\Users\<you>\AppData\Local\Programs\Ollama\ollama.exe`.
4. Open each affected file listed above, search for the three hardcoded values, and replace them with yours. Keep each file's string style exactly as it is:
   - `.py` files use raw strings: `Path(r"D:\EYAD\KitAI")` → `Path(r"C:\Users\you\Projects\KitAI")`
   - `.ps1` files use single quotes: `'C:\Users\abdel\...python.exe'` → `'C:\Users\you\...python.exe'`
   - `.bat` files use double quotes: `"C:\Users\abdel\...python.exe"` → `"C:\Users\you\...python.exe"`
   - `.json` / `.md` files are metadata and documentation — fix them for correctness.

5. **Tip (make it username-proof):** instead of your absolute path you can use portable forms, so the script works on any Windows username:
   - PowerShell: `Join-Path $env:USERPROFILE 'AppData\Local\Programs\Ollama\ollama.exe'`
   - Batch: `%LOCALAPPDATA%\Programs\Ollama\ollama.exe`
   - Python: `Path.home() / "AppData" / "Local" / "Programs" / "Ollama" / "ollama.exe"`

6. Verify nothing is left:

   ```powershell
   Get-ChildItem -Recurse -File |
     Where-Object { $_.FullName -notmatch '\\.git\\' } |
     Select-String -Pattern 'D:\EYAD', 'Users\\abdel'
   ```

   Zero output means you are done.

## Option 2 (easiest) — Let a free agent like OpenCode do it

[OpenCode](https://opencode.ai) is a free, open-source coding agent that runs in your terminal. Install it, `cd` into your cloned copy of this repository, start it with `opencode`, and paste this prompt. Fill in the three values at the top if you know them — or leave them as `TODO` and the agent will try to detect them or ask you.

```text
This repository (KitAI) was developed on the original author's Windows machine
and contains hardcoded absolute paths that must be replaced with the real paths
on MY machine before any of the scripts can run.

My values (fill in; if a line says TODO, detect it or ask me):
- Project root (where this repo is cloned): <PROJECT_ROOT>
- Python interpreter with PyTorch + CUDA:   <PYTHON_EXE>
- Ollama executable:                        <OLLAMA_EXE>

Do exactly this:
1. Search the entire repository (excluding .git) for these hardcoded values:
   - D:\EYAD\KitAI  (also written D:/EYAD/KitAI and D:\\EYAD\\KitAI)
   - C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe
   - C:\Users\abdel\AppData\Local\Programs\Ollama\ollama.exe
2. Replace every occurrence with my matching value above. Preserve the exact
   string style of each file: raw strings in Python (r"..."), single quotes in
   PowerShell, double quotes in .bat, double-backslash escapes in JSON.
3. Where a portable form is natural, prefer it over my absolute username path:
   - .ps1:  Join-Path $env:USERPROFILE 'AppData\Local\Programs\Ollama\ollama.exe'
   - .bat:  %LOCALAPPDATA%\Programs\Ollama\ollama.exe
   - .py:   Path.home() / "AppData" / "Local" / "Programs" / ...
4. Do NOT change anything else: no file renames, no refactors, no README.md
   edits, and leave author-name mentions (pyproject.toml, README citation)
   untouched. Only paths change.
5. When finished, verify: search the repo for "EYAD" and "Users\abdel"
   (case-insensitive, excluding .git) and confirm zero remaining matches.
6. Print a table of every file you changed with the number of replacements
   made in each.
```

After the agent finishes, review its change list, spot-check two or three files, and run the verification command from Option 1 to confirm no hardcoded path remains.
