"""
Splits the total fileset and creates condor job submission files for the specified run script.

Author(s): Cristina Mantilla Suarez, Raghav Kansal
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
from math import ceil
from pathlib import Path
from string import Template

from colorama import Fore, Style

from boostedhh import utils

t2_redirectors = {
    "lpc": "root://cmseos.fnal.gov//",
    "ucsd": "root://redirector.t2.ucsd.edu:1094//",
}

REPO_DICT = {"bbbb": "HH4b", "bbtautau": "bbtautau"}


def print_red(s):
    return print(f"{Fore.RED}{s}{Style.RESET_ALL}")


def write_template(templ_file: str, out_file: str, templ_args: dict):
    """Write to ``out_file`` based on template from ``templ_file`` using ``templ_args``"""

    with Path(templ_file).open() as f:
        templ = Template(f.read())

    with Path(out_file).open("w") as f:
        f.write(templ.substitute(templ_args))


def parse_submit_args(parser):
    parser.add_argument(
        "--analysis", required=True, choices=["bbbb", "bbtautau"], help="which analysis", type=str
    )
    parser.add_argument("--script", default="src/run.py", help="script to run", type=str)
    parser.add_argument("--tag", default="Test", help="process tag", type=str)
    parser.add_argument(
        "--outdir", dest="outdir", default="outfiles", help="directory for output files", type=str
    )
    parser.add_argument(
        "--fileset-path",
        default=None,
        help="path to index .json file. Will use hard-coded defaults per analysis if unspecified.",
        type=str,
    )
    parser.add_argument(
        "--site",
        default="lpc",
        help="computing cluster we're running this on",
        type=str,
        choices=["lpc", "ucsd"],
    )
    parser.add_argument(
        "--save-sites",
        default=["lpc", "ucsd"],
        help="tier 2s in which we want to save the files",
        type=str,
        nargs="+",
        choices=["lpc", "ucsd"],
    )
    utils.add_bool_arg(
        parser,
        "test",
        default=False,
        help="test run or not - test run means only 2 jobs per sample will be created",
    )
    parser.add_argument("--files-per-job", default=20, help="# files per condor job", type=int)
    utils.add_bool_arg(parser, "submit", default=False, help="submit files as well as create them")
    parser.add_argument("--git-branch", required=True, help="git branch to use", type=str)
    parser.add_argument("--git-user", default="LPC-HH", help="which user's repo to use", type=str)
    utils.add_bool_arg(
        parser,
        "allow-diff-local-repo",
        default=False,
        help="Allow the local repo to be different from the specified remote repo (not recommended!)."
        "If false, submit script will exit if the latest commits locally and on Github are different.",
    )


def check_branch(
    analysis: str, git_branch: str, git_user: str = "LPC-HH", allow_diff_local_repo: bool = False
):
    """Check that specified git branch exists in the repo, and local repo is up-to-date"""
    repo = REPO_DICT[analysis]

    assert not bool(
        os.system(
            f'git ls-remote --exit-code --heads "https://github.com/{git_user}/{repo}" "{git_branch}"'
        )
    ), f"Branch {git_branch} does not exist"

    print(f"Using branch {git_branch}")

    # check if there are uncommitted changes
    uncommited_files = int(subprocess.getoutput("git status -s | wc -l"))

    if uncommited_files:
        print_red("There are local changes that have not been committed!")
        os.system("git status -s")
        if allow_diff_local_repo:
            print_red("Proceeding anyway...")
        else:
            print_red("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)

    # check that the local repo's latest commit matches that on github
    remote_hash = subprocess.getoutput(f"git show origin/{git_branch} | head -n 1").split(" ")[1]
    local_hash = subprocess.getoutput("git rev-parse HEAD")

    if remote_hash != local_hash:
        print_red("Latest local and github commits do not match!")
        print(f"Local commit hash: {local_hash}")
        print(f"Remote commit hash: {remote_hash}")
        if allow_diff_local_repo:
            print_red("Proceeding anyway...")
        else:
            print_red("Exiting! Use the --allow-diff-local-repo option to override this.")
            sys.exit(1)


def init_args(args):
    # check that branch exists
    check_branch(args.analysis, args.git_branch, args.git_user, args.allow_diff_local_repo)

    if isinstance(args.year, list):
        if len(args.year) == 1:
            args.year = args.year[0]
        else:
            raise ValueError("Submitting multiple years without --yaml option is not supported yet")

    username = os.environ["USER"]

    if args.site == "lpc":
        try:
            proxy = os.environ["X509_USER_PROXY"]
        except KeyError as e:
            raise FileNotFoundError("No proxy found on LPC. Exiting.") from e
    elif args.site == "ucsd":
        if username == "rkansal":
            proxy = "/home/users/rkansal/x509up_u31735"
        elif username == "dprimosc":
            proxy = "/tmp/x509up_u150012"  # "/home/users/dprimosc/x509up_u150012"
        elif username == "lumori":
            proxy = "/tmp/x509up_u81981"
    else:
        raise ValueError(f"Invalid site {args.site}")

    if args.site not in args.save_sites:
        warnings.warn(
            f"Your local site {args.site} is not in save sites {args.save_sites}!", stacklevel=1
        )

    t2_prefixes = [t2_redirectors[site] for site in args.save_sites]

    tag = f"{args.tag}_{args.nano_version}_{args.region}"

    # make eos dir
    pdir = Path(f"store/user/{username}/{args.analysis}/{args.processor}/")
    outdir = pdir / tag

    # make local directory
    local_dir = Path(f"condor/{args.processor}/{tag}")
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    print("Condor work dir: ", local_dir)

    print("Subsamples", args.subsamples)

    return proxy, t2_prefixes, outdir, local_dir


def submit(
    args,
    proxy,
    t2_prefixes,
    outdir,
    local_dir,
    fileset: dict,
    processor_args: str = "",
):
    """Create condor submission files and optionally submit them"""
    jdl_templ = "boostedhh/condor/submit.templ.jdl"
    sh_templ = "boostedhh/condor/submit.templ.sh"

    # submit jobs
    nsubmit = 0
    for sample, sfiles in fileset.items():
        for subsample, tot_files in sfiles.items():
            if args.submit:
                print("Submitting " + subsample)

            sample_dir = outdir / args.year / subsample
            njobs = ceil(tot_files / args.files_per_job)

            for j in range(njobs):
                if args.test and j == 2:
                    break

                prefix = f"{args.year}_{subsample}"
                localcondor = f"{local_dir}/{prefix}_{j}.jdl"
                jdl_args = {"dir": local_dir, "prefix": prefix, "jobid": j, "proxy": proxy}
                write_template(jdl_templ, localcondor, jdl_args)

                localsh = f"{local_dir}/{prefix}_{j}.sh"
                sh_args = {
                    "repo": REPO_DICT[args.analysis],
                    "branch": args.git_branch,
                    "gituser": args.git_user,
                    "script": args.script,
                    "year": args.year,
                    "starti": j * args.files_per_job,
                    "endi": (j + 1) * args.files_per_job,
                    "batch_size": args.batch_size,
                    "sample": sample,
                    "subsample": subsample,
                    "processor": args.processor,
                    "maxchunks": args.maxchunks,
                    "chunksize": args.chunksize,
                    "t2_prefixes": " ".join(t2_prefixes),
                    "outdir": sample_dir,
                    "filetag": j,
                    "jobnum": j,
                    "save_root": ("--save-root" if args.save_root else "--no-save-root"),
                    "nano_version": args.nano_version,
                    "save_systematics": (
                        "--save-systematics" if args.save_systematics else "--no-save-systematics"
                    ),
                    "processor_args": processor_args,
                }
                write_template(sh_templ, localsh, sh_args)
                os.system(f"chmod u+x {localsh}")

                if Path(f"{localcondor}.log").exists():
                    Path(f"{localcondor}.log").unlink()

                if args.submit:
                    os.system(f"condor_submit {localcondor}")
                else:
                    print("To submit ", localcondor)
                nsubmit = nsubmit + 1

    print(f"Total {nsubmit} jobs")


def replace_batch_size(file_path: Path, new_batch_size: int):
    """Replacing batch size in given file"""
    import re

    # Read the file
    with file_path.open("r") as file:
        content = file.read()

    # Replace using regex
    updated_content = re.sub(r"--batch-size \d+", f"--batch-size {new_batch_size}", content)

    # Write back to the file
    with file_path.open("w") as file:
        file.write(updated_content)
