import concurrent.futures
import os
from pathlib import Path
import subprocess
root = Path(__file__).resolve().parent
jobs = [(d,s) for s in [10000,9000,9500,8000,8500] for d in ['epit','soccol']]
def work(job):
    d,s=job
    with (root/f'{d}_{s}.log').open('w') as log:
        result=subprocess.run(['.venv/bin/python',str(root/'extract.py'),'--dataset',d,'--step',str(s)],
            stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'PYTHONPATH':'src:.','OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
    print(d,s,result.returncode,flush=True)
    return result.returncode
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    codes=list(pool.map(work,jobs))
assert not any(codes),codes
