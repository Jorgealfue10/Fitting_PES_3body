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

def read_output(file):
    with open(file) as f:
        lines = f.readlines()

    return {
        "iterations": parse_iterations(lines),
        "residuals": parse_residuals(lines),
        "summary": parse_summary(lines),
        "vex": parse_vex(lines)
    }


def parse_iterations(lines: list[str]):
    rows = [] ; reading = False

    for line in lines: 
        if ("iteration" in  line and "rms(u.a.)" in line and "Emax(kcal/mol)" in line):
            reading = True
            continue
        if not reading:
            continue
        if "v-inp" in line and "v-fit" in line:
            break

        values = line.split()

        if len(values) != 6:
            continue

        try: 
            iteration = int(values[0])
            numeric_values = [
                float(value.replace("D","E").replace("d","e"))
                for value in values[1:]
            ]
        except ValueError:
            continue

        rows.append([iteration,*numeric_values])

    if not rows:
        raise ValueError("No iterations found in output")

    return pd.DataFrame(
        rows,
        columns=[
            "iteration","vex1","vex2",
            "rms_au","rms_kcal","emax_kcal"
        ]
    )

def parse_residuals(lines: list[str]):
    rows = [] ; reading = False

    for line in lines:
        if ("v-inp" in line and "v-fit" in line and "diff(u.a.)" in line):
            reading = True
            continue
        if not reading:
            continue
        if "n effectif=" in line: 
            break

        values = line.split()

        if len(values) != 7:
            continue

        try:
            numeric_values = [
                float(value.replace("D", "E").replace("d", "e"))
                for value in values
            ]
        except ValueError:
            continue

        rows.append(numeric_values)

    if not rows:
        raise ValueError("Not residuals found")

    return pd.DataFrame(
        rows,
        columns=[
            "r12","r13","r23",
            "v_inp","v_fit",
            "diff_au","diff_kcal"
        ]
    )

def parse_summary(lines: list[str]) -> dict[str, float]:
    summary = {}

    for line in lines:
        if line.strip().startswith("n effectif="):
            values = line.split()
            summary["neff"] = int(values[-1])
        if line.strip().startswith("RMS="):
            values = line.replace("D", "E").split()

            summary["rms_au"] = float(values[-5])
            summary["rms_kcal"] = float(values[-2])

        elif line.strip().startswith("Emax"):
            values = line.replace("D", "E").split()

            summary["emax_kcal"] = float(values[2])

    required = {"neff","rms_au", "rms_kcal", "emax_kcal"}

    if summary.keys() < required:
        raise ValueError("No se pudo leer el summary completo.")

    return summary

def parse_vex(lines: list[str]): 
    vexvals = [] 
    line = lines[-1]
    if line.strip().startswith("vex1(1)="):
        values = line.split()
        vexvals.append(values[1])
        vexvals.append(values[-1])
    return vexvals

