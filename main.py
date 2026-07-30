import numpy as np
import pandas as pd
from pathlib import Path
import subprocess

def read_PES(file: str | Path):
    filename = Path(file)
    pes = np.loadtxt(filename)
    df = pd.DataFrame(
        pes,
        columns=["r12","r13","r23","Eab"]
    )
    return df

def write_inp(
        inp_file: str | Path,
        r12: np.ndarray, r13: np.ndarray, r23: np.ndarray,
        Eab: np.ndarray,
        indx: int, ifor: int, numiter: int, lim: int,
        npnts: int, nordr: int, vex: np.ndarray | list[float], e0: float,
        weights: np.ndarray | None = None
):

    for arr in [r13, r23, Eab]:
        if arr.shape != r12.shape:
            raise ValueError("All arrays must have the same shape.")

    if weights is not None and weights.shape != r12.shape:
        raise ValueError("Weights must have the same shape as the coordinates.")

    with open(inp_file,"w") as f:
        f.write(f"{indx} {ifor} {numiter} {lim} \n")
        f.write(f"{npnts} {nordr}")
        for vxi in vex: 
            f.write(f"{vxi:12.8f} ")
        f.write(f"{lim} \n")

        if weights is None:        
            for i in range(npnts):
                rab = r12[i] ; rac = r13[i] ; rbc = r23[i] ; E = Eab[i]
                f.write(f"{rab:7.3f} {rbc:7.3f} {rac:7.3f} {E:12.8f} \n")
        else:
            for i in range(npnts):
                rab = r12[i] ; rac = r13[i] ; rbc = r23[i] ; E = Eab[i] ; w = weights[i]
                f.write(f"{rab:7.3f} {rbc:7.3f} {rac:7.3f} {E:12.8f} {w:12.8f} \n")

def run_fit(
        exec: str | Path, inp_file: str | Path, workdir: str | Path | None = None,
        timeout: float | None = None
) -> subprocess.CompletedProcess[str]:

    executable = Path(exec).resolve()
    inp_file = Path(inp_file).resolve()
    if workdir is None:
        workdir = inp_file.parent
    workdir = Path(workdir).resolve()

    if not executable.exists():
        raise FileNotFoundError(f"Executable not found: {executable}")
    if not inp_file.exists():
        raise FileNotFoundError(f"Input file not found: {inp_file}")

    with open(inp_file,"r") as f:
        result = subprocess.run(
            [str(executable)],stdin=f,cwd=workdir,
            capture_output=True,text=True,timeout=timeout,check=False
        )

    (workdir / "stdout.log").write_text(result.stdout)
    (workdir / "stderr.log").write_text(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"El ajuste ha fallado con código {result.returncode}.\n"
            f"Revisa:\n"
            f"  {workdir / 'stdout.log'}\n"
            f"  {workdir / 'stderr.log'}"
        )

    return result