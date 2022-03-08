import os
import subprocess
import htcondor


def condor_submit(jobfile):
    """Handle the job submission for HTCondor."""
    cmd = ["condor_submit", jobfile]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"Condor submission failed. Stderr:\n {stderr}.")
    jobid = stdout.split()[-1].decode("utf-8").replace(".", "")
    return jobid
