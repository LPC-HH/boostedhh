"""
Create a JSON list of files of privately produced NanoAOD files.

Author: Raghav Kansal
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

from XRootD import client

from boostedhh import hh_vars, utils


def _dirlist(fs, path) -> list:
    status, listing = fs.dirlist(str(path))
    if not status.ok:
        raise FileNotFoundError(f"Failed to list directory: {status}")

    return [f.name for f in listing]

def _has_new_structure(fs, base_dir, user, years):
    """Check if the directory uses the new structure (data_{year}, mc_{year}) or old structure ({year})."""
    user_path = base_dir / user
    try:
        user_contents = _dirlist(fs, user_path)
    except FileNotFoundError:
        return False
    
    # Check if any data_{year} or mc_{year} directories exist
    for year in years:
        if f"data_{year}" in user_contents or f"mc_{year}" in user_contents:
            return True
    
    return False

def _get_sample_from_subsample(subsample_name, is_data):
    """
    Determine the sample name from the subsample name using the SAMPLES dictionary.
    Source: 
    - https://github.com/rkansal47/Run3_nano_submission/blob/40a74eeffd5d0b935629567dc291a32c9c43abb7/datasets/get_datasets.py
    - https://github.com/rkansal47/Run3_nano_submission/blob/40a74eeffd5d0b935629567dc291a32c9c43abb7/datasets/get_mc.py
    """    
    # If no match found, try to infer from common patterns
    if is_data:
        # Data
        if "JetHT" in subsample_name or "JetMET" in subsample_name:
            return "JetMET"
        elif "EGamma" in subsample_name:
            return "EGamma"
        elif "Muon" in subsample_name:
            return "Muon"
        elif "Tau" in subsample_name:
            return "Tau"
        elif "BTagMu" in subsample_name:
            return "BTagMu"
        elif "MuonEG" in subsample_name:
            return "MuonEG"
        elif "ParkingVBF" in subsample_name:
            return "ParkingVBF"
        elif "ParkingSingleMuon" in subsample_name:
            return "ParkingSingleMuon"
        else:
            raise ValueError(f"Could not determine sample from subsample name: {subsample_name}. Please check the naming conventions.")
    else:
        # MC
        if "HHto4B" in subsample_name or "HHto2B2Tau" in subsample_name:
            if "VBF" in subsample_name:
                return "HHbbtt" if "2B2Tau" in subsample_name else "HH4b"
            else:
                return "HHbbtt" if "2B2Tau" in subsample_name else "HH4b"
        elif "Hto2B" in subsample_name:
            return "Hbb"
        elif "Hto2C" in subsample_name:
            return "Hcc"
        elif "Hto2Tau" in subsample_name or "HTo2Tau" in subsample_name:
            return "Htautau"
        elif "QCD-4Jets_HT" in subsample_name:
            return "QCD"
        elif "QCD_PT" in subsample_name:
            return "QCD_PT"
        elif "TTto" in subsample_name:
            return "TT"
        elif any(x in subsample_name for x in ["TbarWplus", "TWminus", "TbarBQ", "TBbarQ"]):
            return "SingleTop"
        elif "DYto2L-4Jets" in subsample_name:
            return "DYJetsLO"
        elif "DYto2L-2Jets" in subsample_name:
            return "DYJetsNLO"
        elif any(x in subsample_name for x in ["Wto2Q-3Jets", "WtoLNu-4Jets", "Zto2Q-4Jets"]):
            return "VJetsLO"
        elif any(x in subsample_name for x in ["Wto2Q-2Jets", "WtoLNu-2Jets", "Zto2Q-2Jets"]):
            return "VJetsNLO"
        elif any(x in subsample_name for x in ["WW_", "WZ_", "ZZ_", "WWto4Q", "WWtoLNu2Q", "WZto3LNu", "WZto4Q", "ZZto2L2Q", "ZZto4L"]):
            return "Diboson"
        elif any(x in subsample_name for x in ["VBFZto2Q", "VBFWto2Q", "VBFto2L", "VBFto2Nu", "VBFtoLNu"]):
            return "EWKV"
        elif any(x in subsample_name for x in ["WGtoLNuG", "WGto2QG", "ZGto2NuG", "ZGto2QG"]):
            return "VGamma"
    
        raise ValueError(f"Could not determine sample from subsample name: {subsample_name}. Please check the naming conventions.")


def xrootd_index_private_nano(
    base_dir: str,
    redirector: str = "root://cmseos.fnal.gov/",
    users: list[str] = None,
    years: list[str] = None,
    samples: list[str] = None,
    subsamples: list[str] = None,
    files: dict[str] = None,
    overwrite_sample: bool = False,
) -> list:
    """Recursively search for privately produced NanoAOD files via XRootD.

    Can specify specific users, years, samples, and subsamples to search for;
    otherwise, it will search for all by default.

    Supports both old and new directory structures:

    Old structure:
    MC:
    ......redirector.......|...............base_dir....................|..user.|year|sample|
    root://cmseos.fnal.gov//store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022/HHbbtt/

    New structure:
    MC:
    ......redirector.......|...............base_dir....................|..user.|year|sample|
    root://cmseos.fnal.gov//store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022_mc/HHbbtt/

    Data:
    ......redirector.......|...............base_dir....................|..user.|year|sample|
    root://cmseos.fnal.gov//store/user/lpcdihiggsboost/NanoAOD_v12_ParT/rkansal/2022_data/Tau/
    """
    fs = client.FileSystem(redirector)
    base_dir = Path(base_dir)

    users = _dirlist(fs, base_dir) if users is None else users
    years = hh_vars.years if years is None else years

    if files is None:
        files = {}
        
    # Check version
    if len(users) > 0:
        use_new_structure = _has_new_structure(fs, base_dir, users[0], years)
        print(f"Using {'new' if use_new_structure else 'old'} directory structure")
    else:
        # no users to search for
        return {}

    for user in users:
        print(f"\t{user}")
        
        for year in years:
            print(f"\t\t{year}")
            if year not in files:
                files[year] = {}

            if use_new_structure:
                # New structure: separate data_{year} and mc_{year} directories
                for is_data in (True, False):
                    if is_data:
                        ypath = base_dir / user / f"data_{year}"
                    else:
                        ypath = base_dir / user / f"mc_{year}"
                    
                    tsubsamples = _dirlist(fs, ypath) if subsamples is None else subsamples
                        
                    for subsample in tsubsamples:
                        print(f"\t\t\tProcessing {subsample}")
                        sample = _get_sample_from_subsample(subsample, is_data)
                        
                        # Filter by samples if specified
                        if samples is not None and sample not in samples:
                            continue
                            
                        if sample not in files[year]:
                            files[year][sample] = {}
                        elif overwrite_sample:
                            warnings.warn(f"Overwriting existing sample {sample}", stacklevel=2)
                            files[year][sample] = {}

                        print(f"\t\t\t{sample}")
                        spath = ypath / subsample

                        # Clean subsample name
                        subsample_name = subsample.split("_TuneCP5")[0].split("_LHEweights")[0]
                        print(f"\t\t\t\t{subsample_name}")
                        
                        if not is_data:
                            if subsample_name in files[year][sample]:
                                warnings.warn(
                                    f"Duplicate subsample found! {subsample=} ({subsample_name=}) for {year=}",
                                    stacklevel=2
                                )
                            print(f"\t\t\t\t{subsample_name}")

                        # Navigate through the directory structure (4 levels for new structure)
                        tfiles = []
                        try:
                            for f1 in _dirlist(fs, spath):  # dataset directory
                                f1path = spath / f1
                                for f2 in _dirlist(fs, f1path):  # timestamp directory
                                    f2path = f1path / f2
                                    for f3 in _dirlist(fs, f2path):  # chunk directory (0000, 0001, etc.)
                                        f3path = f2path / f3
                                        f3_contents = _dirlist(fs, f3path)
                                        root_files = [f for f in f3_contents if f.endswith(".root")]
                                        if root_files:
                                            tfiles += [f"{redirector}{f3path!s}/{f}" for f in root_files]

                                if is_data:
                                    # For data, concatenate files from related subsamples
                                    # e.g. EGamma0 and EGamma1 should be combined
                                    if f1 not in files[year][sample]:
                                        files[year][sample][f"{sample}_{f1}"] = []
                                    files[year][sample][f"{sample}_{f1}"].extend(tfiles)
                                    print(f"\t\t\t\t\t{len(tfiles)} files added")

                            if not is_data:
                                files[year][sample][subsample_name] = tfiles
                                print(f"\t\t\t\t\t{len(tfiles)} files")
                                
                        except FileNotFoundError:
                            print(f"\t\t\t\tWarning: Could not access {spath}")
                            continue
                            
            else:
                # Old structure: single year directory
                ypath = base_dir / user / year
                try:
                    tsamples = _dirlist(fs, ypath) if samples is None else samples
                except FileNotFoundError:
                    continue
                    
                for sample in tsamples:
                    if sample not in files[year]:
                        files[year][sample] = {}
                    elif overwrite_sample:
                        warnings.warn(f"Overwriting existing sample {sample}", stacklevel=2)
                        files[year][sample] = {}

                    print(f"\t\t\t{sample}")
                    spath = ypath / sample

                    is_data = sample in hh_vars.DATA_SAMPLES

                    try:
                        tsubsamples = _dirlist(fs, spath) if subsamples is None else subsamples
                    except FileNotFoundError:
                        continue
                        
                    for subsample in tsubsamples:
                        subsample_name = subsample.split("_TuneCP5")[0].split("_LHEweights")[0]
                        if not is_data:
                            if subsample_name in files[year][sample]:
                                warnings.warn(
                                    f"Duplicate subsample found! {subsample_name}", stacklevel=2
                                )

                            print(f"\t\t\t\t{subsample_name}")

                        sspath = spath / subsample
                        try:
                            for f1 in _dirlist(fs, sspath):
                                # For Data files, f1 is the subsample name
                                if is_data:
                                    if f1 in files[year][sample]:
                                        warnings.warn(f"Duplicate subsample found! {f1}", stacklevel=2)

                                    print(f"\t\t\t\t{f1}")

                                f1path = sspath / f1
                                for f2 in _dirlist(fs, f1path):
                                    f2path = f1path / f2
                                    tfiles = []
                                    for f3 in _dirlist(fs, f2path):
                                        f3path = f2path / f3
                                        tfiles += [
                                            f"{redirector}{f3path!s}/{f}"
                                            for f in _dirlist(fs, f3path)
                                            if f.endswith(".root")
                                        ]

                                if is_data:
                                    files[year][sample][f1] = tfiles
                                    print(f"\t\t\t\t\t{len(tfiles)} files")

                            if not is_data:
                                files[year][sample][subsample_name] = tfiles
                                print(f"\t\t\t\t\t{len(tfiles)} files")
                                
                        except FileNotFoundError:
                            print(f"\t\t\t\tWarning: Could not access {sspath}")
                            continue

    return files

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--out-name",
        type=str,
        default="index",
        help="Output JSON name (year and .json will automatically be appended)",
    )

    utils.add_bool_arg(
        parser, "append", "Append to existing JSON file versus overwriting it", default=True
    )

    utils.add_bool_arg(
        parser, "overwrite-sample", "Overwrite an existing sample list in the JSON", default=False
    )

    parser.add_argument(
        "--redirector",
        type=str,
        default="root://cmseos.fnal.gov/",
        help="Base XRootD redirector",
    )

    parser.add_argument(
        "--base-dir",
        type=str,
        default="/store/user/lpcdihiggsboost/NanoAOD_v12_ParT",
        help="Base directory for XRootD search",
    )

    parser.add_argument(
        "--users",
        nargs="+",
        type=str,
        help="Which users' directories. By default searches all.",
        default=None,
    )

    parser.add_argument(
        "--years",
        nargs="+",
        type=str,
        help="Which years to index. By default searches all.",
        default=hh_vars.years,
    )

    parser.add_argument(
        "--samples",
        nargs="+",
        type=str,
        help="Which samples to index. By default searches all.",
        default=None,
    )

    parser.add_argument(
        "--subsamples",
        nargs="+",
        type=str,
        help="Which subsamples to index. By default searches all.",
        default=None,
    )

    args = parser.parse_args()

    if args.append:
        # check if output file exists for each year; if so, load and save to files dict.
        files = {}
        for year in args.years:
            try:
                with Path(f"{args.out_name}_{year}.json").open() as f:
                    files[year] = json.load(f)
            except FileNotFoundError:
                continue
    else:
        files = None

    files = xrootd_index_private_nano(
        args.base_dir,
        args.redirector,
        args.users,
        args.years,
        args.samples,
        args.subsamples,
        files,
        args.overwrite_sample,
    )

    # save files per year
    for year in files:
        with Path(f"{args.out_name}_{year}.json").open("w") as f:
            json.dump(files[year], f, indent=4)


if __name__ == "__main__":
    main()
