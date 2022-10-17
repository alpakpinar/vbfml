import os
import socket
import subprocess
import htcondor


def condor_submit(jobfile):
    """Handle the job submission for HTCondor."""
    # Slightly different implementations for lxplus and LPC nodes
    hostname = socket.gethostname()
    if "lxplus" in hostname:
        cmd = ["condor_submit", jobfile]
    elif "fnal" in hostname:
        cmd = ["bash", "/usr/local/bin/condor_submit", jobfile]
    # We only support lxplus and LPC
    else:
        raise RuntimeError(f"Unrecognized host name: {hostname}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    stdout, stderr = proc.communicate()

    # Stop execution if condor submission fails
    if proc.returncode != 0:
        errmsg = stderr.decode("utf-8")
        raise RuntimeError(f"Condor submission failed. Stderr:\n {errmsg}.")

    jobid = stdout.split()[-1].decode("utf-8").replace(".", "")
    return jobid
