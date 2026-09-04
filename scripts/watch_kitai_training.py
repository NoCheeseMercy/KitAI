"""KitAI local watchdog. Dry-run is the default; --apply permits restart."""
from __future__ import annotations
import argparse, json, logging, re, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import os

STEP_RE = re.compile(r"(?:global\s+)?[Ss]tep\s*[:=]?\s*(\d+)")
CKPT_RE = re.compile(r"^checkpoint_step_(\d{8})\.pt$")
MIN_CKPT_BYTES = 2_200_000_000

# Candidate trainer interpreters, best first. The first entry is the verified
# working interpreter on this machine (torch 2.11.0+cu128, CUDA=True,
# confirmed 2026-09-04). The bare uv Pythons under AppData/Roaming have no
# torch installed and must never be used to launch the trainer.
TORCH_PYTHON_CANDIDATES = [
    r'C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe',
]

PROBE_CODE = "import torch, sys; print('CUDA=' + str(torch.cuda.is_available())); print('TORCH=' + torch.__version__)"


def find_torch_python(log):
    """Return the first candidate interpreter with working torch+CUDA, else None."""
    seen = []
    for raw in TORCH_PYTHON_CANDIDATES + [sys.executable]:
        try:
            cand = str(Path(raw).resolve())
        except OSError:
            continue
        if cand in seen:
            continue
        seen.append(cand)
        if not Path(cand).is_file():
            log.warning('Interpreter candidate not found: %s', cand)
            continue
        try:
            r = subprocess.run([cand, '-c', PROBE_CODE], capture_output=True,
                               text=True, timeout=120, encoding='utf-8', errors='replace')
        except Exception as exc:
            log.warning('Interpreter probe failed for %s: %s', cand, exc)
            continue
        out = (r.stdout or '')
        if r.returncode == 0 and 'CUDA=True' in out:
            torch_ver = next((l for l in out.splitlines() if l.startswith('TORCH=')), 'TORCH=?')
            log.info('Selected trainer interpreter %s (%s)', cand, torch_ver)
            return Path(cand)
        log.warning('Interpreter lacks torch/CUDA, skipping: %s :: %s%s',
                    cand, out.strip().splitlines()[:2], (r.stderr or '').strip().splitlines()[:2])
    return None

def now(): return datetime.now(timezone.utc).isoformat()

def args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root', type=Path, default=root)
    p.add_argument('--checkpoint-dir', type=Path, default=root/'checkpoints/kitai_200m_scratch_pretrain_protected_rerun')
    p.add_argument('--log-file', type=Path, default=root/'logs/kitai_watchdog.log')
    p.add_argument('--poll-seconds', type=int, default=60)
    p.add_argument('--stall-minutes', type=int, default=45)
    p.add_argument('--once', action='store_true')
    p.add_argument('--apply', action='store_true', help='Allow stopping/restarting after confirmed stall')
    return p.parse_args()

def matching_trainers():
    cmd = ['powershell.exe','-NoProfile','-Command',
      "$x=Get-CimInstance Win32_Process -Filter \"Name = 'python.exe'\" | Where-Object {$_.CommandLine -match 'scripts\\.train' -and $_.CommandLine -match 'kitai_200m_scratch_pretrain_protected_rerun\\.yaml'} | Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress; if($x){$x}"]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True, encoding='utf-8', errors='replace').stdout.strip()
        if not out: return []
        x=json.loads(out); return [x] if isinstance(x,dict) else x
    except Exception: return []

def newest_log(root):
    files=list((root/'logs').glob('**/training_stdout*.log'))
    return max(files,key=lambda p:p.stat().st_mtime) if files else None

def latest_step(log):
    if not log or not log.is_file(): return None,None
    try:
        st=log.stat(); text=log.read_text(encoding='utf-8',errors='replace')
        m=list(STEP_RE.finditer(text)); return (int(m[-1].group(1)) if m else None),st.st_mtime
    except OSError: return None,None

def newest_checkpoint(folder):
    good=[]
    for p in folder.glob('checkpoint_step_*.pt'):
        m=CKPT_RE.match(p.name)
        if not m or p.stat().st_size<MIN_CKPT_BYTES: continue
        step=int(m.group(1)); metrics=folder/f'metrics_step_{step:08d}.json'
        if metrics.is_file() and metrics.stat().st_size>0: good.append((step,p,metrics))
    return max(good) if good else None

def launch(a, ckpt, step, log):
    data=sorted((a.project_root/'datasets/kitai_200m_scratch/pretrain_fineweb_edu').glob('fineweb_edu_*.txt'))
    if len(data)!=15: raise RuntimeError(f'Expected 15 FineWeb shards, found {len(data)}')
    config=a.project_root/'configs/kitai_200m_scratch_pretrain_protected_rerun.yaml'
    validation=a.project_root/'datasets/kitai_200m_scratch/validation_report.json'
    if not config.is_file() or not validation.is_file() or not json.loads(validation.read_text()).get('passed'): raise RuntimeError('Protected config/validation missing or failed')
    py = find_torch_python(log)
    if py is None:
        raise RuntimeError('No interpreter with working torch+CUDA found; refusing to restart trainer.')
    audit=a.project_root/'logs/kitai_watchdog'/f'recovery_from_{step:08d}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    audit.mkdir(parents=True,exist_ok=False)
    out,err=audit/'training_stdout.log',audit/'training_stderr.log'
    argv=['-u','-m','scripts.train','--config',str(config),'--data',','.join(map(str,data[:14])),'--val-data',str(data[-1]),'--dataset-kind','streaming_document','--resume',str(ckpt)]
    (audit/'resume_manifest.json').write_text(json.dumps({'launched_at':now(),'source_checkpoint':str(ckpt),'source_step':step,'mode':'full-state --resume','arguments':argv},indent=2))
    # Use environment variables to avoid PowerShell argument slicing/quoting issues.
    env = dict(os.environ)
    env['KITAI_PYTHON'] = str(py)
    env['KITAI_ARGS'] = subprocess.list2cmdline(argv)
    env['KITAI_ROOT'] = str(a.project_root)
    env['KITAI_OUT'] = str(out)
    env['KITAI_ERR'] = str(err)
    ps='$p=Start-Process -FilePath $env:KITAI_PYTHON -ArgumentList $env:KITAI_ARGS -WorkingDirectory $env:KITAI_ROOT -WindowStyle Hidden -RedirectStandardOutput $env:KITAI_OUT -RedirectStandardError $env:KITAI_ERR -PassThru; $p.Id'
    r=subprocess.run(['powershell.exe','-NoProfile','-Command',ps],env=env,capture_output=True,text=True,check=True)
    pid=int(r.stdout.strip().splitlines()[-1]); log.warning('Restarted protected trainer PID %s from step %s',pid,step); return pid

def main():
    a=args(); a.project_root=a.project_root.resolve(); a.checkpoint_dir=a.checkpoint_dir.resolve(); a.log_file=a.log_file.resolve(); a.log_file.parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s | %(levelname)s | %(message)s',handlers=[logging.FileHandler(a.log_file,encoding='utf-8'),logging.StreamHandler()]); log=logging.getLogger('kitai_watchdog')
    log.info('Started; dry_run=%s; stall_minutes=%s',not a.apply,a.stall_minutes)
    while True:
        procs=matching_trainers(); step,mtime=latest_step(newest_log(a.project_root)); ck=newest_checkpoint(a.checkpoint_dir); log_stale=mtime is None or time.time()-mtime>=a.stall_minutes*60
        stalled=(len(procs)==0) or log_stale
        log.info('status trainer_count=%s report_step=%s log_age=%s durable_step=%s stalled=%s',len(procs),step,None if mtime is None else round(time.time()-mtime),None if not ck else ck[0],stalled)
        if stalled and a.apply and len(procs)<=1 and ck:
            if procs: subprocess.run(['powershell.exe','-NoProfile','-Command',f'Stop-Process -Id {int(procs[0]["ProcessId"])} -Force'],check=True); time.sleep(5)
            launch(a,ck[1],ck[0],log)
            if a.once: return
        if a.once: return
        time.sleep(max(10,a.poll_seconds))
if __name__=='__main__': main()
