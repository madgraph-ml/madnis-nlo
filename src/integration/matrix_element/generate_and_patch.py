#!/usr/bin/env python3
"""
Generate and patch MadGraph matrix element APIs for e+e- -> 3 jets and 4 jets.

Follows the procedure documented in README.md:
  Born         -> patch_born.sh
  Loop         -> patch_loop.sh
  Counterterms -> patch_soft_integrated.sh
                  patch_collinear_integrated.sh
                  patch_real.sh

Usage:
  python generate_and_patch.py [--mg5-dir /path/to/mg5amcnlo]
"""

import argparse
import re
import subprocess
import sys
import shutil
import tempfile
from pathlib import Path

REQUIRED_COMMIT = "fb1c2f2d067f63cebe6799f582f81cb89157e291"
MG5_REPO_URL = "https://github.com/mg5amcnlo/mg5amcnlo"

# ── helpers ──────────────────────────────────────────────────────────────────


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command, printing it first."""
    cwd_str = str(cwd) if cwd else None
    print(f"\n>>> {cmd}" + (f"  [cwd={cwd}]" if cwd else ""))
    result = subprocess.run(cmd, shell=True, cwd=cwd_str)
    if check and result.returncode != 0:
        print(f"ERROR: command failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def ensure_mg5(mg5_dir_arg):
    """
    Ensure MadGraph5 is available at the required commit.

    If mg5_dir_arg is given it must be an existing git clone;
    otherwise the repo is cloned into ``./mg5amcnlo``.

    Returns (mg5_exe, mg5_dir).
    """
    if mg5_dir_arg:
        mg5_dir = Path(mg5_dir_arg).resolve()
    else:
        mg5_dir = Path.cwd() / "mg5amcnlo"
        if not mg5_dir.exists():
            print(f"Cloning MadGraph5 into {mg5_dir} ...")
            run_cmd(f"git clone {MG5_REPO_URL} {mg5_dir}")

    if not mg5_dir.is_dir():
        print(f"ERROR: {mg5_dir} is not a directory")
        sys.exit(1)

    print(f"Checking out commit {REQUIRED_COMMIT} ...")
    run_cmd(f"git checkout {REQUIRED_COMMIT}", cwd=mg5_dir)

    mg5_exe = mg5_dir / "bin" / "mg5_aMC"
    if not mg5_exe.exists():
        print(f"ERROR: mg5_aMC not found at {mg5_exe}")
        sys.exit(1)

    return mg5_exe, mg5_dir


def run_mg5(mg5_exe, cmd_lines, check=True):
    """Write *cmd_lines* to a temp file and run ``mg5_aMC`` on it.

    MG5 is run inside a temporary directory so that debris files
    (additional_command, ME5_debug, MG5_debug, nsqso_born.inc, ...)
    do not pollute the caller's working directory.
    """
    print("MG5 command file:")
    for line in cmd_lines:
        print(f"  {line}")
    with tempfile.TemporaryDirectory(prefix="mg5_run_") as tmpdir:
        cmdfile = Path(tmpdir) / "mg5_cmd.txt"
        cmdfile.write_text("\n".join(cmd_lines) + "\n")
        result = run_cmd(f"{mg5_exe} {cmdfile}", cwd=tmpdir, check=check)
    return result


# ── card editing ─────────────────────────────────────────────────────────────


def edit_param_card(card_path, as_line):
    """Replace the ``aS`` line in *param_card.dat*."""
    card = Path(card_path)
    if not card.exists():
        print(f"  WARNING: {card} not found, skipping param_card edit")
        return
    text = card.read_text()
    new_text = re.sub(
        r"^\s*3\s+\S+\s*#\s*aS.*$",
        f"    {as_line}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if new_text == text:
        print(f"  WARNING: aS line not matched in {card}")
    card.write_text(new_text)
    print(f"  param_card: aS -> {as_line}")


def edit_run_card(card_path, updates):
    """
    Update run-card parameters in place.

    *updates*: ``{param_name: new_value_string}``.
    Only parameters already present in the file are touched.
    """
    card = Path(card_path)
    if not card.exists():
        print(f"  WARNING: {card} not found, skipping run_card edit")
        return
    text = card.read_text()
    for param, value in updates.items():
        new_text = re.sub(
            rf"^(\s*)\S+(\s*=\s*{re.escape(param)}\b.*)$",
            rf"\g<1>{value}\2",
            text,
            count=1,
            flags=re.MULTILINE,
        )
        if new_text != text:
            text = new_text
            print(f"  run_card: {param} = {value}")
    card.write_text(text)


# ── subprocess discovery ─────────────────────────────────────────────────────


def find_subprocess_dirs(subproc_base):
    """Return sorted list of ``P*`` directories under *subproc_base*."""
    base = Path(subproc_base)
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("P"))

# ── make_opts patching ───────────────────────────────────────────────────────


def patch_make_opts(mg5_dir):
    """Append -fPIC flags to Template/NLO/Source/make_opts.inc if not present."""
    make_opts = Path(mg5_dir) / "Template" / "NLO" / "Source" / "make_opts.inc"
    if not make_opts.exists():
        print(f"  WARNING: {make_opts} not found, skipping -fPIC patch")
        return
    text = make_opts.read_text()
    additions = []
    for flag_line in ("FFLAGS += -fPIC", "CFLAGS += -fPIC", "CXXFLAGS += -fPIC"):
        var = flag_line.split()[0]
        # match both "+= -fPIC" and "+=-fPIC" variants
        if not re.search(rf"^{var}\s*\+=\s*-fPIC", text, re.MULTILINE):
            additions.append(flag_line)
    if additions:
        make_opts.write_text(text.rstrip("\n") + "\n" + "\n".join(additions) + "\n")
        print(f"  make_opts.inc: added {', '.join(additions)}")
    else:
        print("  make_opts.inc: -fPIC flags already present, nothing to do")


# ── process generators ───────────────────────────────────────────────────────

JET_CARD_SETTINGS = {
    "jetalgo": "1.0",
    "jetradius": "0.4",
    "ptj": "20.0",
    "etaj": "5.0",
}


def generate_born(mg5_exe, me_dir, storage_dir, n_jets):
    """
    Born matrix element  (README -- "Born matrix element" section).

    generate e+ e- > u u~ g(g)
    output standalone .../ee_{n}j/born
    launch -f
    edit cards, then patch_born.sh on each subprocess.
    """
    finals = {3: "u u~ g", 4: "u u~ g g"}[n_jets]
    make_dir = storage_dir / f"ee_{n_jets}j"
    make_dir.mkdir(parents=True, exist_ok=True)
    output_dir = make_dir / "born"
    shutil.rmtree(output_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"  BORN  e+e- -> {n_jets} jets")
    print(f"{'=' * 60}")

    # -- MG5: generate, output standalone, compile --------------------------
    # NB: do NOT run mg5 with cwd=me_dir -- there is a local
    # multiprocessing.py that shadows the stdlib module.
    run_mg5(
        mg5_exe,
        [
            f"generate e+ e- > {finals}",
            f"output standalone {output_dir}",
            "launch -f",
        ],
    )

    # -- edit cards ---------------------------------------------------------
    edit_param_card(
        output_dir / "Cards" / "param_card.dat",
        "3 0.11901442560187821 # aS",
    )
    edit_run_card(output_dir / "Cards" / "run_card.dat", JET_CARD_SETTINGS)

    # -- patch each subprocess ----------------------------------------------
    subprocs = find_subprocess_dirs(output_dir / "SubProcesses")
    if not subprocs:
        print(f"ERROR: no P* directories found in {output_dir / 'SubProcesses'}")
        sys.exit(1)

    for sp in subprocs:
        rel = sp.relative_to(me_dir)
        print(f"\n  patch_born.sh {rel}")
        run_cmd(f"bash ./patch_born.sh {rel}", cwd=me_dir)
        if (sp / "api.so").exists():
            print(f"  OK  {sp / 'api.so'}")
        else:
            print(f"  WARNING: api.so not created in {sp}")


def generate_loop(mg5_exe, me_dir, storage_dir, n_jets):
    """
    Loop (virtual) matrix element  (README -- "Loop matrix element" section).

    generate e+ e- > u u~ g(g) [virt=QCD]
    output .../ee_{n}j/loop
    launch -f
    edit cards, then patch_loop.sh on each subprocess.
    """
    finals = {3: "u u~ g", 4: "u u~ g g"}[n_jets]
    make_dir = storage_dir / f"ee_{n_jets}j"
    make_dir.mkdir(parents=True, exist_ok=True)
    output_dir = make_dir / "loop"
    shutil.rmtree(output_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"  LOOP  e+e- -> {n_jets} jets")
    print(f"{'=' * 60}")

    run_mg5(
        mg5_exe,
        [
            f"generate e+ e- > {finals} [virt=QCD]",
            f"output {output_dir}",
            "launch -f",
        ],
    )

    edit_param_card(
        output_dir / "Cards" / "param_card.dat",
        "3 0.11901442560187821 # aS",
    )
    edit_run_card(output_dir / "Cards" / "run_card.dat", JET_CARD_SETTINGS)

    subprocs = find_subprocess_dirs(output_dir / "SubProcesses")
    if not subprocs:
        print(f"ERROR: no P* directories found in {output_dir / 'SubProcesses'}")
        sys.exit(1)

    for sp in subprocs:
        rel = sp.relative_to(me_dir)
        print(f"\n  patch_loop.sh {rel}")
        run_cmd(f"bash ./patch_loop.sh {rel}", cwd=me_dir)
        if (sp / "api.so").exists():
            print(f"  OK  {sp / 'api.so'}")
        else:
            print(f"  WARNING: api.so not created in {sp}")


def generate_counterterms(mg5_exe, me_dir, storage_dir, n_jets):
    """
    NLO counterterms  (README -- Soft / Collinear / Real sections).

    generate e+ e- > j j j(j) [QCD]
    output .../ee_{n}j/counterterms
    compile FO  (sets fixed_order=ON, writes analyse_opts, compiles FO executables)
    edit cards, then patch_soft / patch_collinear / patch_real on each subprocess.
    """
    jets = " ".join(["j"] * n_jets)
    make_dir = storage_dir / f"ee_{n_jets}j"
    make_dir.mkdir(parents=True, exist_ok=True)
    output_dir = make_dir / "counterterms"
    shutil.rmtree(output_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"  COUNTERTERMS  e+e- -> {n_jets} jets [QCD]")
    print(f"{'=' * 60}")

    # Step 1: generate and output the process.
    run_mg5(
        mg5_exe,
        [
            f"generate e+ e- > {jets} [QCD]",
            f"output {output_dir}",
        ],
    )

    # Step 2: compile in FO mode via the process's own aMCatNLO binary.
    # 'compile FO' sets fixed_order=ON, writes analyse_opts, and builds
    # the FO executables needed by the patch scripts -- without running.
    amcatnlo_exe = output_dir / "bin" / "aMCatNLO"
    run_mg5(amcatnlo_exe, ["compile FO"])

    # -- edit cards (note: aS = 0.119, different from born / loop) ----------
    edit_param_card(
        output_dir / "Cards" / "param_card.dat",
        "3 0.119 # aS",
    )
    edit_run_card(output_dir / "Cards" / "run_card.dat", JET_CARD_SETTINGS)

    # -- patch each subprocess with soft, collinear, real -------------------
    subprocs = find_subprocess_dirs(output_dir / "SubProcesses")
    if not subprocs:
        print(f"ERROR: no P* directories found in {output_dir / 'SubProcesses'}")
        sys.exit(1)

    for sp in subprocs:
        rel = sp.relative_to(me_dir)
        print(f"\n  Patching subprocess {sp.name}")

        print(f"    patch_soft_integrated.sh {rel}")
        run_cmd(f"bash ./patch_soft_integrated.sh {rel}", cwd=me_dir)

        print(f"    patch_collinear_integrated.sh {rel}")
        run_cmd(f"bash ./patch_collinear_integrated.sh {rel}", cwd=me_dir)

        print(f"    patch_real.sh {rel}")
        run_cmd(f"bash ./patch_real.sh {rel}", cwd=me_dir)

        for so_name in [
            "api_soft_integrated.so",
            "api_collinear_integrated.so",
            "api_sreal.so",
        ]:
            if (sp / so_name).exists():
                print(f"    OK  {so_name}")
            else:
                print(f"    WARNING: {so_name} not created")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate and patch MadGraph matrix element APIs "
            "for e+e- -> 3 jets and 4 jets."
        )
    )
    parser.add_argument(
        "--mg5-dir",
        default=None,
        help=(
            "Path to an existing MadGraph5 git repository.  "
            "If omitted, mg5amcnlo is cloned into the current directory."
        ),
    )
    args = parser.parse_args()

    # Resolve key directories
    me_dir = Path(__file__).resolve().parent  # .../src/integration/matrix_element
    storage_dir = me_dir / "process_api_storage"
    storage_dir.mkdir(parents=True, exist_ok=True)

    mg5_exe, mg5_dir = ensure_mg5(args.mg5_dir)
    patch_make_opts(mg5_dir)
    print(f"MadGraph5 dir : {mg5_dir}")
    print(f"matrix_element: {me_dir}")
    print(f"storage       : {storage_dir}")

    for n_jets in [3, 4]:
        generate_born(mg5_exe, me_dir, storage_dir, n_jets)
        generate_loop(mg5_exe, me_dir, storage_dir, n_jets)
        generate_counterterms(mg5_exe, me_dir, storage_dir, n_jets)

    print(f"\n{'=' * 60}")
    print("All processes generated and patched.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
