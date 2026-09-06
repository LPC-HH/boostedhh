"""
Collection of utilities for corrections and systematics in processors.

Loosely based on https://github.com/jennetd/hbb-coffea/blob/master/boostedhiggs/corrections.py

Most corrections retrieved from the cms-nanoAOD repo:
See https://cms-nanoaod-integration.web.cern.ch/commonJSONSFs/

Authors: Raghav Kansal, Cristina Suarez
"""

from __future__ import annotations

import gzip
import pathlib
import pickle
import random
import sys
import json

import awkward as ak
import correctionlib
import numpy as np
import uproot
from coffea.analysis_tools import Weights
from coffea.nanoevents.methods import vector
from coffea.nanoevents.methods.base import NanoEventsArray
from coffea.nanoevents.methods.nanoaod import FatJetArray, JetArray

from .utils import pad_val


ak.behavior.update(vector.behavior)
package_path = str(pathlib.Path(__file__).parent.parent.resolve())

# Important Run3 start of Run
FirstRun_2022C = 355794
FirstRun_2022D = 357487
LastRun_2022D = 359021
FirstRun_2022E = 359022
LastRun_2022F = 362180

"""
CorrectionLib files are available from: /cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration - synced daily
"""
pog_correction_path = "/cvmfs/cms.cern.ch/rsync/cms-nanoAOD/jsonpog-integration/"
# jin-start
pog_correction_path_update = "/cvmfs/cms-griddata.cern.ch/cat/metadata/"
# jin-end
pog_jsons = {
    "muon": ["MUO", "muon_Z.json.gz"],
    "muon_scale": ["MUO", "muon_scalesmearing.json.gz"],
    "electron": ["EGM", "electron.json.gz"],
    "electron_hlt": ["EGM", "electronHlt.json.gz"],
    "electron_ss": ["EGM", "electronSS_EtDependent.json.gz"],
    "pileup": ["LUM", "puWeights.json.gz"],
    "fatjet_jec": ["JME", "fatJet_jerc.json.gz"],
    "jet_jec": ["JME", "jet_jerc.json.gz"],
    "jetveto": ["JME", "jetvetomaps.json.gz"],
    "btagging": ["BTV", "btagging.json.gz"],
    "tau": ["TAU", "tau.json.gz"],
}

# compatibility with old correctionlib:
# convert string infinity edges to floating-point infinity
def _replace_inf(obj):
    if isinstance(obj, dict):
        return {k: _replace_inf(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace_inf(v) for v in obj]
    if obj in ["+inf", "inf"]:
        return float("inf")
    if obj == "-inf":
        return float("-inf")
    return obj

def _get_category_value(node, key):
    if node["nodetype"] == "transform":
        transformed_key = next(
            item["value"]
            for item in node["rule"]["content"]
            if item["key"] == key
        )
        node = node["content"]
        key = transformed_key

    return next(
        item["value"]
        for item in node["content"]
        if item["key"] == key
    )

def get_UL_year(year: str) -> str:
    return f"{year}_UL"

# jin-start
def get_pog_json_update(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = get_UL_year(year) if year == "2018" else year
    original_year = year
    if "2022" in year or "2023" in year or "2024" in year:
        year = {
            "2022": "Run3-22CDSep23-Summer22-NanoAODv12",
            "2022EE": "Run3-22EFGSep23-Summer22EE-NanoAODv12",
            "2023": "Run3-23CSep23-Summer23-NanoAODv12",
            "2023BPix": "Run3-23DSep23-Summer23BPix-NanoAODv12",
            "2024": "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15",
        }[year]
    if original_year == "2024" and obj == "pileup":
        return f"{pog_correction_path_update}/{pog_json[0]}/{year}/latest/puWeights_CDEFGHI.json.gz"
    return f"{pog_correction_path_update}/{pog_json[0]}/{year}/latest/{pog_json[1]}"

def add_pileup_weight_update(weights: Weights, year: str, nPU: np.ndarray, dataset: str | None = None):
    # clip nPU from 0 to 100
    nPU = np.clip(nPU, 0, 99)
    # print(list(nPU))

    if "Pu60" in dataset or "Pu70" in dataset:
        # pileup profile from data
        path_pileup = package_path + "/corrections/data/MyDataPileupHistogram2022FG.root"
        pileup_profile = uproot.open(path_pileup)["pileup"]
        pileup_profile = pileup_profile.to_numpy()[0]
        # normalise
        pileup_profile /= pileup_profile.sum()

        # https://indico.cern.ch/event/695872/contributions/2877123/attachments/1593469/2522749/pileup_ppd_feb_2018.pdf
        # pileup profile from MC
        pu_name = "Pu60" if "Pu60" in dataset else "Pu70"
        path_pileup_dataset = package_path + f"/corrections/data/pileup/{pu_name}.npy"
        pileup_MC = np.load(path_pileup_dataset)

        # avoid division by 0 (?)
        pileup_MC[pileup_MC == 0.0] = 1
        pileup_correction = pileup_profile / pileup_MC
        # remove large MC reweighting factors to prevent artifacts
        pileup_correction[pileup_correction > 10] = 10
        sf = pileup_correction[nPU]
        # no uncertainties
        weights.add("pileup", sf)

    else:

        # https://cms-analysis-corrections.docs.cern.ch/corrections_era/
        values = {}
        
        cset = correctionlib.CorrectionSet.from_file(get_pog_json_update("pileup", year))
        corr = {
            "2018": "Collisions18_UltraLegacy_goldenJSON",
            "2022": "Collisions2022_355100_357900_eraBCD_GoldenJson",
            "2022EE": "Collisions2022_359022_362760_eraEFG_GoldenJson",
            "2023": "Collisions2023_366403_369802_eraBC_GoldenJson",
            "2023BPix": "Collisions2023_369803_370790_eraD_GoldenJson",
            "2024": "Collisions24_CDEFGHI_goldenJSON",
        }[year]
        # evaluate and clip up to 10 to avoid large weights
        values["nominal"] = np.clip(cset[corr].evaluate(nPU, "nominal"), 0, 10)
        values["up"] = np.clip(cset[corr].evaluate(nPU, "up"), 0, 10)
        values["down"] = np.clip(cset[corr].evaluate(nPU, "down"), 0, 10)

        weights.add("pileup", values["nominal"], values["up"], values["down"])


def get_tau_tes(taus, year: str, isData: bool = False, wp: str = "Medium", wp_VSe: str = "VVLoose"):
    """
    Apply Tau Energy Scale (TES) correction.

    Currently only applied for 2022 MC.

    Returns
    -------
    taus_corrected:
        Tau collection with nominal TES applied to pt and mass.

    tes_shifted_vars:
        Dictionary containing nominal / TES up / TES down
        pt and mass values.
    """

    if isData:
        return taus, None
    # if year != "2022":
    #     return taus, None

    path = get_pog_json_update("tau", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload["corrections"] = [
        correction
        for correction in payload["corrections"]
        if correction["name"] == "tau_energy_scale"
    ]

    cset = correctionlib.CorrectionSet.from_string(
        json.dumps(payload)
    )

    corr = cset["tau_energy_scale"]

    # Tau is a jagged collection:
    # save number of taus in each event,
    # then flatten before correctionlib
    counts = ak.num(taus, axis=1)
    flat_taus = ak.flatten(taus, axis=1)

    if len(flat_taus) == 0:
        return taus, None

    pt = ak.to_numpy(flat_taus.pt)
    eta = ak.to_numpy(flat_taus.eta)
    dm = ak.to_numpy(flat_taus.decayMode)
    genmatch = ak.to_numpy(flat_taus.genPartFlav)

    # Evaluate TES nominal / up / down
    def evaluate_tes(syst):

        # Default: no TES correction
        # Objects not covered by the official payload keep TES = 1
        tes_flat = np.ones(len(pt), dtype=float)

        # Genuine hadronic tau:
        # TES is defined for DM = 0, 1, 10, 11
        genuine_tau = (
            (genmatch == 5)
            & np.isin(dm, [0, 1, 10, 11])
        )

        # Electron -> tau fake
        if year == "2024":
            electron_fake_dms = [0, 1, 10, 11]
        else:
            electron_fake_dms = [0, 1, 2, 10, 11]

        electron_fake = (
            np.isin(genmatch, [1, 3])
            & np.isin(dm, electron_fake_dms)
        )

        # Muon -> tau fake:
        # correction does not depend on decay mode
        muon_fake = np.isin(genmatch, [2, 4])

        # Only evaluate objects covered by the payload
        valid = genuine_tau | electron_fake | muon_fake

        if np.any(valid):
            tes_flat[valid] = corr.evaluate(
                pt[valid],
                eta[valid],
                dm[valid],
                genmatch[valid],
                "DeepTau2018v2p5",
                wp,
                wp_VSe,
                syst,
            )

        return ak.unflatten(tes_flat, counts)


    tes_nom = evaluate_tes("nom")
    tes_up = evaluate_tes("up")
    tes_down = evaluate_tes("down")

    # Apply NOMINAL TES to Tau four-vector
    taus_corrected = ak.with_field(
        taus,
        taus.pt * tes_nom,
        "pt",
    )

    taus_corrected = ak.with_field(
        taus_corrected,
        taus.mass * tes_nom,
        "mass",
    )

    # Keep Up/Down for future systematics
    tes_shifted_vars = {
        "pt": {
            "": taus.pt * tes_nom,
            "TES_up": taus.pt * tes_up,
            "TES_down": taus.pt * tes_down,
        },
        "mass": {
            "": taus.mass * tes_nom,
            "TES_up": taus.mass * tes_up,
            "TES_down": taus.mass * tes_down,
        },
    }

    return taus_corrected, tes_shifted_vars

def get_tau_vsjet_sf(taus, year: str, wp: str = "Medium", wp_VSe: str = "VVLoose", flag: str = "pt"):
    """
    Get DeepTau2018v2p5VSjet scale factors for selected taus.

    Returns per-tau nominal/up/down SFs.
    Objects not covered by the SF are assigned SF = 1.
    """

    # if year != "2022":
    #     return None

    path = get_pog_json_update("tau", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload["corrections"] = [
        correction
        for correction in payload["corrections"]
        if correction["name"] == "DeepTau2018v2p5VSjet"
    ]

    cset = correctionlib.CorrectionSet.from_string(
        json.dumps(payload)
    )

    corr = cset["DeepTau2018v2p5VSjet"]

    counts = ak.num(taus, axis=1)
    flat_taus = ak.flatten(taus, axis=1)

    if len(flat_taus) == 0:
        ones = ak.ones_like(taus.pt)
        return {
            "nom": ones,
            "up": ones,
            "down": ones,
        }

    pt = ak.to_numpy(flat_taus.pt)
    dm = ak.to_numpy(flat_taus.decayMode)
    genmatch = ak.to_numpy(flat_taus.genPartFlav)

    def evaluate_sf(syst):

        # Default = 1.
        # VSjet SF applies to genuine taus only.
        sf_flat = np.ones(len(pt), dtype=float)

        valid = (
            (genmatch == 5)
            & np.isin(dm, [0, 1, 10, 11])
        )

        low_pt = valid & (pt < 140.0)
        high_pt = valid & (pt >= 140.0)

        if np.any(low_pt):
            sf_flat[low_pt] = corr.evaluate(
                pt[low_pt],
                dm[low_pt],
                genmatch[low_pt],
                wp,
                wp_VSe,
                syst,
                "dm",
            )

        if np.any(high_pt):
            sf_flat[high_pt] = corr.evaluate(
                pt[high_pt],
                dm[high_pt],
                genmatch[high_pt],
                wp,
                wp_VSe,
                syst,
                "pt",
            )

        return ak.unflatten(sf_flat, counts)

    return {
        "nom": evaluate_sf("nom"),
        "up": evaluate_sf("up"),
        "down": evaluate_sf("down"),
    }

def get_tau_trigger_sf(taus, year: str, trigtype: str, wp: str = "Medium", corrtype: str = "sf"):
    """Get per-tau tau-trigger nominal/up/down corrections."""

    # if year != "2022":
    #     return None

    path = get_pog_json_update("tau", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload["corrections"] = [correction for correction in payload["corrections"] if correction["name"] == "tau_trigger"]

    node = payload["corrections"][0]["data"]
    for key in [trigtype, corrtype, wp, -1, "nom"]:
        node = _get_category_value(node, key)
    pt_min = float(node["edges"][0])

    payload = _replace_inf(payload)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    corr = cset["tau_trigger"]

    counts = ak.num(taus, axis=1)
    flat_taus = ak.flatten(taus, axis=1)

    if len(flat_taus) == 0:
        ones = ak.ones_like(taus.pt)
        return {"nom": ones, "up": ones, "down": ones}

    pt = ak.to_numpy(flat_taus.pt)
    dm = np.full(len(pt), -1, dtype=int)

    def evaluate_sf(syst):
        sf_flat = np.ones(len(pt), dtype=float)
        valid = pt >= pt_min
        if np.any(valid):
            sf_flat[valid] = corr.evaluate(pt[valid], dm[valid], trigtype, wp, corrtype, syst)
        return ak.unflatten(sf_flat, counts)

    return {"nom": evaluate_sf("nom"), "up": evaluate_sf("up"), "down": evaluate_sf("down")}


from .MuonScaRe import pt_scale, pt_resol, pt_scale_var, pt_resol_var

def get_muon_scale_smearing(events, muons, year, isData):
    cset = correctionlib.CorrectionSet.from_file(get_pog_json_update("muon_scale", year))

    pt_scaled = pt_scale(
        isData, muons.pt, muons.eta, muons.phi, muons.charge, cset, nested=True
    )

    if isData:
        pt_nom = pt_scaled
        muon_shifted_vars = {"pt": {"": pt_nom}}
    else:
        pt_nom = pt_resol(
            pt_scaled, muons.eta, muons.phi, muons.nTrackerLayers,
            events.event, events.luminosityBlock, cset, nested=True,
        )

        muon_shifted_vars = {
            "pt": {
                "": pt_nom,
                "MuonScale_up": pt_scale_var(
                    pt_nom, muons.eta, muons.phi, muons.charge, "up", cset, nested=True
                ),
                "MuonScale_down": pt_scale_var(
                    pt_nom, muons.eta, muons.phi, muons.charge, "dn", cset, nested=True
                ),
                "MuonReso_up": pt_resol_var(
                    pt_scaled, pt_nom, muons.eta, "up", cset, nested=True
                ),
                "MuonReso_down": pt_resol_var(
                    pt_scaled, pt_nom, muons.eta, "dn", cset, nested=True
                ),
            }
        }

    muons_corrected = ak.with_field(muons, pt_nom, "pt")
    return muons_corrected, muon_shifted_vars

def get_muon_id_sfs(muons, year):
    """
    Muon TightID SF.
    """

    path = get_pog_json_update("muon", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload["corrections"] = [
        correction for correction in payload["corrections"]
        if correction["name"] == "NUM_TightID_DEN_TrackerMuons"
    ]

    payload = _replace_inf(payload)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    corr = cset["NUM_TightID_DEN_TrackerMuons"]

    counts = ak.num(muons)
    pt = ak.to_numpy(ak.flatten(muons.pt))
    # Pass the original signed eta:
    # 2022/2022EE JSON applies abs(eta) internally;
    # 2023/2023BPix/2024 use signed-eta bins.
    eta = ak.to_numpy(ak.flatten(muons.eta))

    def evaluate_sf(variation):
        variation_map = {
            "nom": "nominal",
            "up": "systup",
            "down": "systdown",
        }

        sf = corr.evaluate(
            eta,
            pt,
            variation_map[variation],
        )

        return ak.unflatten(sf, counts)

    return {
        "nom": evaluate_sf("nom"),
        "up": evaluate_sf("up"),
        "down": evaluate_sf("down"),
    }

def get_muon_trigger_sfs(muons, year):

    path = get_pog_json_update("muon", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    triggers = [
        correction for correction in payload["corrections"]
        if "_DEN_CutBasedId" in correction["name"]
    ]
    payload["corrections"] = triggers

    payload = _replace_inf(payload)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))

    counts = ak.num(muons)
    pt = ak.to_numpy(ak.flatten(muons.pt))
    eta = ak.to_numpy(ak.flatten(muons.eta))

    variation_map = {
        "nom": "nominal",
        "up": "systup",
        "down": "systdown",
    }

    output = {}

    for trigger in triggers:
        name = trigger["name"]
        corr = cset[name]

        eta_min, eta_max = trigger["data"]["edges"][0], trigger["data"]["edges"][-1]
        pt_min = trigger["data"]["content"][0]["edges"][0]
        valid = (eta >= eta_min) & (eta < eta_max) & (pt >= pt_min)

        def evaluate_sf(variation):
            sf = np.ones_like(pt, dtype=float)
            sf[valid] = corr.evaluate(eta[valid], pt[valid], variation_map[variation])
            return ak.unflatten(sf, counts)

        output[name] = {
            "nom": evaluate_sf("nom"),
            "up": evaluate_sf("up"),
            "down": evaluate_sf("down"),
        }

    return output
        
def get_electron_scale_smearing(events, electrons, year, isData):
    """
    Apply Run-3 electron energy scale / smearing corrections.

    Data:
        apply nominal energy scale correction.

    MC:
        apply nominal smearing and provide scale/smearing variations.
    """

    path = get_pog_json_update("electron_ss", year)
    cset = correctionlib.CorrectionSet.from_file(path)

    counts = ak.num(electrons, axis=1)
    flat_electrons = ak.flatten(electrons, axis=1)

    if len(flat_electrons) == 0:
        return electrons, None

    pt = ak.to_numpy(flat_electrons.pt)
    sceta = ak.to_numpy(flat_electrons.eta + flat_electrons.deltaEtaSC)
    r9 = ak.to_numpy(flat_electrons.r9)

    if isData:
        seed_gain = ak.to_numpy(flat_electrons.seedGain).astype(float)
        run = np.repeat(ak.to_numpy(events.run), ak.to_numpy(counts)).astype(float)

        corr = cset.compound["Scale"]
        scale = corr.evaluate("scale", run, sceta, r9, pt, seed_gain)

        pt_nom = ak.unflatten(pt * scale, counts)

        electron_shifted_vars = {
            "pt": {
                "": pt_nom,
            }
        }

    else:
        corr = cset["SmearAndSyst"]

        smear = corr.evaluate("smear", pt, r9, sceta)
        smear_up = corr.evaluate("smear_up", pt, r9, sceta)
        smear_down = corr.evaluate("smear_down", pt, r9, sceta)
        scale_unc = corr.evaluate("escale", pt, r9, sceta)

        import hashlib

        metadata = events.metadata
        seed_key = f"{metadata.get('fileuuid', metadata.get('filename', ''))}-{metadata.get('entrystart', 0)}-{metadata.get('entrystop', len(events))}-ElectronScaleSmearing"
        seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:8], 16)

        rng = np.random.default_rng(seed)
        rnd = rng.normal(0.0, 1.0, len(pt))

        pt_nom = ak.unflatten(pt * (1.0 + smear * rnd), counts)
        pt_smear_up = ak.unflatten(pt * (1.0 + smear_up * rnd), counts)
        pt_smear_down = ak.unflatten(pt * (1.0 + smear_down * rnd), counts)
        pt_scale_up = ak.unflatten(pt * (1.0 + scale_unc), counts)
        pt_scale_down = ak.unflatten(pt * (1.0 - scale_unc), counts)

        electron_shifted_vars = {
            "pt": {
                "": pt_nom,
                "ElectronScale_up": pt_scale_up,
                "ElectronScale_down": pt_scale_down,
                "ElectronSmear_up": pt_smear_up,
                "ElectronSmear_down": pt_smear_down,
            }
        }

    electrons_corrected = ak.with_field(electrons, pt_nom, "pt")

    return electrons_corrected, electron_shifted_vars

def get_electron_sf_era(year):
    return {"2022": "2022Re-recoBCD", "2022EE": "2022Re-recoE+PromptFG", "2023": "2023PromptC", "2023BPix": "2023PromptD", "2024": "2024Prompt"}[year]

def get_electron_reco_sfs(electrons, year):

    path = get_pog_json_update("electron", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload = _replace_inf(payload)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    corr = cset["Electron-ID-SF"]

    counts = ak.num(electrons)
    pt = ak.to_numpy(ak.flatten(electrons.pt))
    eta = ak.to_numpy(ak.flatten(electrons.eta))
    phi = ak.to_numpy(ak.flatten(electrons.phi))
    era = get_electron_sf_era(year)

    def evaluate_sf(variation):
        variation_map = {"nom": "sf", "up": "sfup", "down": "sfdown"}
        sf = np.ones_like(pt, dtype=float)

        reco_bins = [
            ("RecoBelow20", (pt >= 10) & (pt < 20)),
            ("Reco20to75", (pt >= 20) & (pt < 75)),
            ("RecoAbove75", pt >= 75),
        ]

        for wp, valid in reco_bins:
            if np.any(valid):
                # sf[valid] = corr.evaluate(era, variation_map[variation], wp, eta[valid], pt[valid])
                if year in ("2023", "2023BPix"):
                    sf[valid] = corr.evaluate(era, variation_map[variation], wp, eta[valid], pt[valid], phi[valid])
                else:
                    sf[valid] = corr.evaluate(era, variation_map[variation], wp, eta[valid], pt[valid])

        return ak.unflatten(sf, counts)

    return {"nom": evaluate_sf("nom"), "up": evaluate_sf("up"), "down": evaluate_sf("down")}

def get_electron_id_sfs(electrons, year, wp="wp90noiso"):

    path = get_pog_json_update("electron", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload = _replace_inf(payload)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    corr = cset["Electron-ID-SF"]

    counts = ak.num(electrons)
    pt = ak.to_numpy(ak.flatten(electrons.pt))
    eta = ak.to_numpy(ak.flatten(electrons.eta))
    phi = ak.to_numpy(ak.flatten(electrons.phi))
    era = get_electron_sf_era(year)

    variation_map = {"nom": "sf", "up": "sfup", "down": "sfdown"}

    def evaluate_sf(variation):
        sf = np.ones_like(pt, dtype=float)
        valid = pt >= 10
        if np.any(valid):
            # sf[valid] = corr.evaluate(era, variation_map[variation], wp, eta[valid], pt[valid])
            if year in ("2023", "2023BPix"):
                sf[valid] = corr.evaluate(era, variation_map[variation], wp, eta[valid], pt[valid], phi[valid])
            else:
                sf[valid] = corr.evaluate(era, variation_map[variation], wp, eta[valid], pt[valid])

        return ak.unflatten(sf, counts)

    return {"nom": evaluate_sf("nom"), "up": evaluate_sf("up"), "down": evaluate_sf("down")}

def get_electron_trigger_sfs(electrons, year, path_name="HLT_SF_Ele30_TightID"):

    path = get_pog_json_update("electron_hlt", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload = _replace_inf(payload)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    corr = cset["Electron-HLT-SF"]

    counts = ak.num(electrons)
    pt = ak.to_numpy(ak.flatten(electrons.pt))
    eta = ak.to_numpy(ak.flatten(electrons.eta))
    era = get_electron_sf_era(year)

    variation_map = {"nom": "sf", "up": "sfup", "down": "sfdown"}

    def evaluate_sf(variation):
        sf = np.ones_like(pt, dtype=float)
        valid = pt >= 25
        if np.any(valid):
            sf[valid] = corr.evaluate(era, variation_map[variation], path_name, eta[valid], pt[valid])
        return ak.unflatten(sf, counts)

    return {"nom": evaluate_sf("nom"), "up": evaluate_sf("up"), "down": evaluate_sf("down")}

def get_btag_sfs(jets, year: str):
    """ParticleNet AK4 b-tagging shape SFs."""

    shape_sf_years = {"2022", "2022EE", "2023", "2023BPix"}

    if year not in shape_sf_years:
        return None

    path = get_pog_json_update("btagging", year)

    with gzip.open(path, "rt") as f:
        payload = json.load(f)

    payload["corrections"] = [c for c in payload["corrections"] if c["name"] == "particleNet_shape"]
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    corr = cset["particleNet_shape"]

    counts = ak.num(jets, axis=1)
    flat_jets = ak.flatten(jets, axis=1)

    if len(flat_jets) == 0:
        ones = ak.ones_like(jets.pt)
        return {key: ones for key in ["nom", "hf_up", "hf_down", "lf_up", "lf_down", "hfstats1_up", "hfstats1_down", "hfstats2_up", "hfstats2_down", "lfstats1_up", "lfstats1_down", "lfstats2_up", "lfstats2_down", "cferr1_up", "cferr1_down", "cferr2_up", "cferr2_down"]}

    pt = ak.to_numpy(flat_jets.pt)
    eta = np.abs(ak.to_numpy(flat_jets.eta))
    flavor = ak.to_numpy(flat_jets.hadronFlavour).astype(np.int64)
    score = ak.to_numpy(flat_jets.btagPNetB)

    valid = (eta < 2.5) & np.isin(flavor, [0, 4, 5]) & (score >= 0.0) & (score < 1.1)

    sf_nom = np.ones(len(pt))
    sf_nom[valid] = corr.evaluate("central", flavor[valid], eta[valid], pt[valid], score[valid])

    sfs = {"nom": ak.unflatten(sf_nom, counts)}

    for source in ["hf", "lf", "hfstats1", "hfstats2", "lfstats1", "lfstats2"]:
        mask = valid & np.isin(flavor, [0, 5])
        for direction in ["up", "down"]:
            sf = sf_nom.copy()
            sf[mask] = corr.evaluate(f"{direction}_{source}", flavor[mask], eta[mask], pt[mask], score[mask])
            sfs[f"{source}_{direction}"] = ak.unflatten(sf, counts)

    for source in ["cferr1", "cferr2"]:
        mask = valid & (flavor == 4)
        for direction in ["up", "down"]:
            sf = sf_nom.copy()
            sf[mask] = corr.evaluate(f"{direction}_{source}", flavor[mask], eta[mask], pt[mask], score[mask])
            sfs[f"{source}_{direction}"] = ak.unflatten(sf, counts)

    return sfs

def get_pog_json(obj: str, year: str) -> str:
    try:
        pog_json = pog_jsons[obj]
    except:
        print(f"No json for {obj}")

    year = get_UL_year(year) if year == "2018" else year
    if "2022" in year or "2023" in year or "2024" in year:
        year = {
            "2022": "2022_Summer22",
            "2022EE": "2022_Summer22EE",
            "2023": "2023_Summer23",
            "2023BPix": "2023_Summer23BPix",
            "2024": "2024_Summer24",
        }[year]
    return f"{pog_correction_path}/POG/{pog_json[0]}/{year}/{pog_json[1]}"


def add_pileup_weight(weights: Weights, year: str, nPU: np.ndarray, dataset: str | None = None):
    # clip nPU from 0 to 100
    nPU = np.clip(nPU, 0, 99)
    # print(list(nPU))

    if "Pu60" in dataset or "Pu70" in dataset:
        # pileup profile from data
        path_pileup = package_path + "/corrections/data/MyDataPileupHistogram2022FG.root"
        pileup_profile = uproot.open(path_pileup)["pileup"]
        pileup_profile = pileup_profile.to_numpy()[0]
        # normalise
        pileup_profile /= pileup_profile.sum()

        # https://indico.cern.ch/event/695872/contributions/2877123/attachments/1593469/2522749/pileup_ppd_feb_2018.pdf
        # pileup profile from MC
        pu_name = "Pu60" if "Pu60" in dataset else "Pu70"
        path_pileup_dataset = package_path + f"/corrections/data/pileup/{pu_name}.npy"
        pileup_MC = np.load(path_pileup_dataset)

        # avoid division by 0 (?)
        pileup_MC[pileup_MC == 0.0] = 1
        pileup_correction = pileup_profile / pileup_MC
        # remove large MC reweighting factors to prevent artifacts
        pileup_correction[pileup_correction > 10] = 10
        sf = pileup_correction[nPU]
        # no uncertainties
        weights.add("pileup", sf)

    else:
        
        if year == "2024":
            import json
            
            # Read directly from the Golden JSON file
            golden_json_path = "/afs/cern.ch/user/j/jinwa/Cert_Collisions2024_378981_386951_Golden.json"
            
            print(f"Loading pileup from Golden JSON: {golden_json_path}")
            
            # For 2024, temporarily use a nominal weight of 1 (if there is no pileup information in the Golden JSON).
            values = {}
            values["nominal"] = np.ones_like(nPU, dtype=float)
            values["up"] = np.ones_like(nPU, dtype=float)
            values["down"] = np.ones_like(nPU, dtype=float)
            
            print(f"Using default pileup weights (all 1.0) for 2024")
            
            weights.add("pileup", values["nominal"], values["up"], values["down"])
        
        else:

            # https://twiki.cern.ch/twiki/bin/view/CMS/LumiRecommendationsRun3
            values = {}
            
            cset = correctionlib.CorrectionSet.from_file(get_pog_json("pileup", year))
            corr = {
                "2018": "Collisions18_UltraLegacy_goldenJSON",
                "2022": "Collisions2022_355100_357900_eraBCD_GoldenJson",
                "2022EE": "Collisions2022_359022_362760_eraEFG_GoldenJson",
                "2023": "Collisions2023_366403_369802_eraBC_GoldenJson",
                "2023BPix": "Collisions2023_369803_370790_eraD_GoldenJson",
            }[year]
            # evaluate and clip up to 10 to avoid large weights
            values["nominal"] = np.clip(cset[corr].evaluate(nPU, "nominal"), 0, 10)
            values["up"] = np.clip(cset[corr].evaluate(nPU, "up"), 0, 10)
            values["down"] = np.clip(cset[corr].evaluate(nPU, "down"), 0, 10)

            weights.add("pileup", values["nominal"], values["up"], values["down"])


def get_vpt(genpart, check_offshell=False):
    """Only the leptonic samples have no resonance in the decay tree, and only
    when M is beyond the configured Breit-Wigner cutoff (usually 15*width)
    """
    boson = ak.firsts(
        genpart[
            ((genpart.pdgId == 23) | (abs(genpart.pdgId) == 24))
            & genpart.hasFlags(["fromHardProcess", "isLastCopy"])
        ]
    )
    if check_offshell:
        offshell = genpart[
            genpart.hasFlags(["fromHardProcess", "isLastCopy"])
            & ak.is_none(boson)
            & (abs(genpart.pdgId) >= 11)
            & (abs(genpart.pdgId) <= 16)
        ].sum()
        return ak.where(ak.is_none(boson.pt), offshell.pt, boson.pt)
    return np.array(ak.fill_none(boson.pt, 0.0))


def add_ps_weight(weights, ps_weights):
    """
    Parton Shower Weights (FSR and ISR)
    """

    nweights = len(weights.weight())
    nom = np.ones(nweights)

    up_isr = np.ones(nweights)
    down_isr = np.ones(nweights)
    up_fsr = np.ones(nweights)
    down_fsr = np.ones(nweights)

    if len(ps_weights[0]) == 4:
        up_isr = ps_weights[:, 0]  # ISR=2, FSR=1
        down_isr = ps_weights[:, 2]  # ISR=0.5, FSR=1

        up_fsr = ps_weights[:, 1]  # ISR=1, FSR=2
        down_fsr = ps_weights[:, 3]  # ISR=1, FSR=0.5

    elif len(ps_weights[0]) > 1:
        print("PS weight vector has length ", len(ps_weights[0]))

    weights.add("ISRPartonShower", nom, up_isr, down_isr)
    weights.add("FSRPartonShower", nom, up_fsr, down_fsr)


def get_pdf_weights(events):
    """
    For the PDF acceptance uncertainty:
        - store 103 variations. 0-100 PDF values
        - The last two values: alpha_s variations.
        - you just sum the yield difference from the nominal in quadrature to get the total uncertainty.
        e.g. https://github.com/LPC-HH/HHLooper/blob/master/python/prepare_card_SR_final.py#L258
        and https://github.com/LPC-HH/HHLooper/blob/master/app/HHLooper.cc#L1488

    Some references:
    Scale/PDF weights in MC https://twiki.cern.ch/twiki/bin/view/CMS/HowToPDF
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#PDF
    """
    return events.LHEPdfWeight.to_numpy()


def get_scale_weights(events):
    """
    QCD Scale variations, best explanation I found is here:
    https://twiki.cern.ch/twiki/bin/viewauth/CMS/TopSystematics#Factorization_and_renormalizatio

    TLDR: we want to vary the renormalization and factorization scales by a factor of 0.5 and 2,
    and then take the envelope of the variations on our final observation as the up/down uncertainties.

    Importantly, we need to keep track of the normalization for each variation,
    so that this uncertainty takes into account the acceptance effects of our selections.

    LHE scale variation weights (w_var / w_nominal) (from https://cms-nanoaod-integration.web.cern.ch/autoDoc/NanoAODv9/2018UL/doc_TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8_RunIISummer20UL18NanoAODv9-106X_upgrade2018_realistic_v16_L1v1-v1.html#LHEScaleWeight)
    [0] is renscfact=0.5d0 facscfact=0.5d0 ; <=
    [1] is renscfact=0.5d0 facscfact=1d0 ; <=
    [2] is renscfact=0.5d0 facscfact=2d0 ;
    [3] is renscfact=1d0 facscfact=0.5d0 ; <=
    [4] is renscfact=1d0 facscfact=1d0 ;
    [5] is renscfact=1d0 facscfact=2d0 ; <=
    [6] is renscfact=2d0 facscfact=0.5d0 ;
    [7] is renscfact=2d0 facscfact=1d0 ; <=
    [8] is renscfact=2d0 facscfact=2d0 ; <=

    See also https://git.rwth-aachen.de/3pia/cms_analyses/common/-/blob/11e0c5225416a580d27718997a11dc3f1ec1e8d1/processor/generator.py#L93 for an example.
    """
    if len(events[0].LHEScaleWeight) > 0:
        if len(events[0].LHEScaleWeight) == 9:
            variations = events.LHEScaleWeight[:, [0, 1, 3, 5, 7, 8]].to_numpy()
            nominal = events.LHEScaleWeight[:, 4].to_numpy()[:, np.newaxis]
            variations /= nominal
        else:
            variations = events.LHEScaleWeight[:, [0, 1, 3, 4, 6, 7]].to_numpy()
        return np.clip(variations, 0.0, 4.0)
    else:
        return None


JER_2024_PREFIX = "Summer24Prompt24_JRV2_MC"

# The official Summer24 V5 data residuals change at these run boundaries.
# Each value is the first run for the corresponding factory stored in
# jec_compiled_update.pkl.gz.
JEC_2024_DATA_IOVS = (
    (378971, "2024_run378971"),
    (380253, "2024_run380253"),
    (380948, "2024_run380948"),
    (381944, "2024_run381944"),
    (382298, "2024_run382298"),
    (383247, "2024_run383247"),
    (383780, "2024_run383780"),
    (384933, "2024_run384933"),
    (385814, "2024_run385814"),
    (386409, "2024_run386409"),
)

# Per-worker cache only.
# This avoids reloading the official JER JSON for every chunk inside one Condor job.
# It is not shared across users, Condor jobs, or Python processes.
_JER_2024_CORRECTION_CACHE = {}

def _get_2024_jer_corrections(fatjets: bool):
    """Load the three official continuous 2024 JER corrections."""

    cache_key = "ak8" if fatjets else "ak4"
    if cache_key in _JER_2024_CORRECTION_CACHE:
        return _JER_2024_CORRECTION_CACHE[cache_key]

    algo = "AK8PFPuppi" if fatjets else "AK4PFPuppi"
    names = {
        kind: f"{JER_2024_PREFIX}_{kind}_{algo}"
        for kind in ("PtResolution", "ScaleFactor", "SFUncertainty")
    }
    path = get_pog_json_update("fatjet_jec" if fatjets else "jet_jec", "2024")

    # Keep only the needed records so the old correctionlib used by this
    # analysis does not have to parse unrelated corrections in the JME file.
    with gzip.open(path, "rt") as handle:
        payload = json.load(handle)

    available = {correction["name"] for correction in payload["corrections"]}
    missing = set(names.values()) - available
    if missing:
        raise KeyError(
            "The 2024 JER payload is missing: " + ", ".join(sorted(missing))
        )

    payload["corrections"] = [
        correction
        for correction in payload["corrections"]
        if correction["name"] in names.values()
    ]
    payload.pop("compound_corrections", None)
    payload = _replace_inf(payload)
    cset = correctionlib.CorrectionSet.from_string(json.dumps(payload))
    corrections = {kind: cset[name] for kind, name in names.items()}

    _JER_2024_CORRECTION_CACHE[cache_key] = corrections
    return corrections


def _evaluate_jer_correction(correction, *, eta, pt, rho):
    """Evaluate a JER record using the input order stored in the JSON."""

    values = {"JetEta": eta, "JetPt": pt, "Rho": rho}
    try:
        arguments = [values[input_.name] for input_ in correction.inputs]
    except KeyError as exc:
        raise KeyError(
            f"Unsupported input {exc.args[0]!r} in JER correction "
            f"{correction.name!r}"
        ) from exc
    return np.asarray(correction.evaluate(*arguments), dtype=np.float64)


def _splitmix64(values):
    """Vectorised SplitMix64 used to obtain reproducible random numbers."""

    values = np.asarray(values, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = values + np.uint64(0x9E3779B97F4A7C15)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(
            0xBF58476D1CE4E5B9
        )
        values = (values ^ (values >> np.uint64(27))) * np.uint64(
            0x94D049BB133111EB
        )
    return values ^ (values >> np.uint64(31))


def _jer_random_normal(events, jets, seed: int):
    """Return one deterministic Gaussian per jet, stable under chunking."""

    event_index = (
        events.event if "event" in events.fields else ak.local_index(events, axis=0)
    )
    run = events.run if "run" in events.fields else ak.zeros_like(event_index)
    lumi = (
        events.luminosityBlock
        if "luminosityBlock" in events.fields
        else ak.zeros_like(event_index)
    )
    jet_index = ak.local_index(jets.pt, axis=1)

    def broadcast_flat(values):
        return ak.to_numpy(
            ak.flatten(ak.broadcast_arrays(values, jets.pt)[0], axis=1)
        ).astype(np.uint64, copy=False)

    event_flat = broadcast_flat(event_index)
    run_flat = broadcast_flat(run)
    lumi_flat = broadcast_flat(lumi)
    jet_flat = ak.to_numpy(ak.flatten(jet_index, axis=1)).astype(
        np.uint64, copy=False
    )

    with np.errstate(over="ignore"):
        key = event_flat.copy()
        key ^= run_flat * np.uint64(0xD2B74407B1CE6E93)
        key ^= lumi_flat * np.uint64(0xCA5A826395121157)
        key ^= jet_flat * np.uint64(0x9E3779B97F4A7C15)
        key ^= np.uint64(seed)

    # Box-Muller transformation. The half-bin offset prevents log(0).
    uniform1 = ((_splitmix64(key) >> np.uint64(11)).astype(np.float64) + 0.5) * (
        1.0 / 2**53
    )
    uniform2 = (
        (_splitmix64(key ^ np.uint64(0xA0761D6478BD642F)) >> np.uint64(11)).astype(
            np.float64
        )
        + 0.5
    ) * (1.0 / 2**53)
    return np.sqrt(-2.0 * np.log(uniform1)) * np.cos(2.0 * np.pi * uniform2)


def get_2024_jer(
    events: NanoEventsArray,
    jets: JetArray,
    *,
    fatjets: bool = False,
    seed: int = 42,
    include_variations: bool = True,
):
    """Apply continuous official 2024 JER after the pkl JEC/JES correction.

    A matched gen jet uses the deterministic hybrid prescription. Unmatched
    jets use stochastic smearing. Nominal/up/down and repeated JES evaluations
    share the same event/jet random number, keeping the shifts correlated.
    """

    counts = ak.to_numpy(ak.num(jets.pt, axis=1))
    if counts.sum() == 0:
        shifted = {variable: {"": jets[variable]} for variable in ("pt", "mass")}
        if include_variations:
            for variable in shifted:
                shifted[variable]["JER_up"] = jets[variable]
                shifted[variable]["JER_down"] = jets[variable]
        return jets, shifted

    pt = ak.to_numpy(ak.flatten(jets.pt, axis=1)).astype(np.float64)
    mass = ak.to_numpy(ak.flatten(jets.mass, axis=1)).astype(np.float64)
    eta = ak.to_numpy(ak.flatten(jets.eta, axis=1)).astype(np.float64)
    phi = ak.to_numpy(ak.flatten(jets.phi, axis=1)).astype(np.float64)
    rho = ak.to_numpy(ak.flatten(jets.event_rho, axis=1)).astype(np.float64)

    corrections = _get_2024_jer_corrections(fatjets)
    resolution = _evaluate_jer_correction(
        corrections["PtResolution"], eta=eta, pt=pt, rho=rho
    )
    sf_nominal = _evaluate_jer_correction(
        corrections["ScaleFactor"], eta=eta, pt=pt, rho=rho
    )
    sf_uncertainty = np.abs(
        _evaluate_jer_correction(
            corrections["SFUncertainty"], eta=eta, pt=pt, rho=rho
        )
    )

    gen = jets.matched_gen
    gen_pt = ak.to_numpy(ak.flatten(ak.fill_none(gen.pt, np.nan), axis=1)).astype(
        np.float64
    )
    gen_eta = ak.to_numpy(
        ak.flatten(ak.fill_none(gen.eta, np.nan), axis=1)
    ).astype(np.float64)
    gen_phi = ak.to_numpy(
        ak.flatten(ak.fill_none(gen.phi, np.nan), axis=1)
    ).astype(np.float64)

    delta_phi = (phi - gen_phi + np.pi) % (2.0 * np.pi) - np.pi
    delta_r = np.hypot(eta - gen_eta, delta_phi)
    radius = 0.8 if fatjets else 0.4
    safe_pt = np.maximum(pt, np.finfo(np.float64).tiny)
    matched = (
        np.isfinite(gen_pt)
        & (gen_pt > 0.0)
        & (delta_r < radius / 2.0)
        & (np.abs(pt - gen_pt) < 3.0 * resolution * pt)
    )

    # Use independent streams for AK4 and AK8, but exactly the same stream for
    # every variation and re-evaluation of a given collection.
    random_normal = _jer_random_normal(
        events, jets, seed + (1_000_003 if fatjets else 0)
    )

    def smear_factor(scale_factor):
        matched_factor = 1.0 + (scale_factor - 1.0) * (pt - gen_pt) / safe_pt
        stochastic_width = resolution * np.sqrt(
            np.maximum(scale_factor * scale_factor - 1.0, 0.0)
        )
        stochastic_factor = 1.0 + random_normal * stochastic_width
        factor = np.where(matched, matched_factor, stochastic_factor)

        # Protect against negative smeared energies, following the usual JER
        # prescription used by the legacy Coffea implementation.
        return np.maximum(factor, 1.0e-2 / safe_pt)

    scale_factors = {"": sf_nominal}
    if include_variations:
        scale_factors.update(
            {
                "JER_up": sf_nominal + sf_uncertainty,
                "JER_down": np.maximum(sf_nominal - sf_uncertainty, 0.0),
            }
        )

    shifted = {"pt": {}, "mass": {}}
    for label, scale_factor in scale_factors.items():
        factor = smear_factor(scale_factor)
        shifted["pt"][label] = ak.unflatten(pt * factor, counts)
        shifted["mass"][label] = ak.unflatten(mass * factor, counts)

    corrected = ak.with_field(jets, shifted["pt"][""], "pt")
    corrected = ak.with_field(corrected, shifted["mass"][""], "mass")
    return corrected, shifted


class JECs:
    def __init__(self, year, jec_compiled=None):
        """Load the JEC/JES factories used by the skimmer.

        The Python-3.11 ``jec_compiled_update.pkl.gz`` contains all Run-3
        factories. It includes legacy JER for 2022--2023BPix, while 2024 JER
        is evaluated directly from correctionlib after applying its JEC/JES.
        """

        if jec_compiled is not None:
            jec_compiled = str(jec_compiled)

        if year in ["2022", "2022EE", "2023", "2023BPix", "2024"]:
            if jec_compiled is None:
                jec_compiled = (
                    package_path + "/corrections/jec_compiled_update.pkl.gz"
                )
        elif year in ["2016", "2016APV", "2017", "2018"]:
            if jec_compiled is None:
                jec_compiled = package_path + "/corrections/jec_compiled_run2.pkl.gz"
        else:
            jec_compiled = None

        self.jet_factory = {}
        self.met_factory = None

        if jec_compiled is not None and pathlib.Path(jec_compiled).is_file():
            print(jec_compiled)
            with gzip.open(jec_compiled, "rb") as filehandler:
                jmestuff = pickle.load(filehandler)

            self.jet_factory["ak4"] = jmestuff["jet_factory"]
            self.jet_factory["ak8"] = jmestuff["fatjet_factory"]
            self.met_factory = jmestuff["met_factory"]
        elif jec_compiled is not None:
            print(f"Warning: JEC/JES pkl not found: {jec_compiled}")

    def _add_jec_variables(self, jets: JetArray, event_rho: ak.Array, isData: bool) -> JetArray:
        """add variables needed for JECs"""
        jets["pt_raw"] = (1 - jets.rawFactor) * jets.pt
        jets["mass_raw"] = (1 - jets.rawFactor) * jets.mass
        jets["event_rho"] = ak.broadcast_arrays(event_rho, jets.pt)[0]
        if not isData:
            jets["pt_gen"] = ak.values_astype(ak.fill_none(jets.matched_gen.pt, 0), np.float32)
        return jets

    def _build_2024_data_jets(self, events, jets, jet_factory_str):
        """Apply the correct Summer24 V5 data factory to every event.

        A chunk may contain more than one official JEC interval. Every factory
        needed by the chunk is evaluated, the corresponding event subset is
        selected, and the original event order is restored afterwards.
        """

        if "run" not in events.fields:
            raise KeyError("2024 data JEC routing requires the events.run branch")

        runs = ak.to_numpy(events.run).astype(np.int64, copy=False)
        if len(runs) != len(jets):
            raise ValueError(
                "events.run and the jet collection have different event counts"
            )
        if len(runs) == 0:
            return jets

        starts = np.asarray([start for start, _ in JEC_2024_DATA_IOVS])
        iov_indices = np.searchsorted(starts, runs, side="right") - 1
        if np.any(iov_indices < 0):
            bad_runs = np.unique(runs[iov_indices < 0])
            raise ValueError(
                "No 2024 data JEC IOV for run(s): "
                + ", ".join(map(str, bad_runs[:10]))
            )

        factories = self.jet_factory[jet_factory_str]
        used_iovs = np.unique(iov_indices)
        used_keys = [JEC_2024_DATA_IOVS[index][1] for index in used_iovs]
        missing = [key for key in used_keys if key not in factories]
        if missing:
            raise KeyError(
                "Missing 2024 data JEC factory key(s): " + ", ".join(missing)
            )

        import cachetools

        if len(used_iovs) == 1:
            key = JEC_2024_DATA_IOVS[used_iovs[0]][1]
            return factories[key].build(jets, cachetools.Cache(np.inf))

        # Evaluate one full corrected collection per IOV present in the chunk.
        # This also keeps a uniform Awkward record type for events with no jets.
        corrected_parts = []
        original_positions = []
        for index in used_iovs:
            mask = iov_indices == index
            key = JEC_2024_DATA_IOVS[index][1]
            corrected_all = factories[key].build(
                jets, cachetools.Cache(np.inf)
            )
            corrected_parts.append(corrected_all[mask])
            original_positions.append(np.nonzero(mask)[0])

        combined = ak.concatenate(corrected_parts, axis=0)
        positions = np.concatenate(original_positions)
        return combined[np.argsort(positions, kind="stable")]

    def get_jec_jets(
        self,
        events: NanoEventsArray,
        jets: FatJetArray,
        year: str,
        isData: bool = False,
        jecs: dict[str, str] | None = None,
        fatjets: bool = True,
        applyData: bool = False,
        dataset: str | None = None,
        nano_version: str = "v12",
    ) -> FatJetArray:
        """Apply nominal JEC/JER and optionally return pt/mass variations."""

        rho = (
            events.Rho.fixedGridRhoFastjetAll
            if "Rho" in events.fields
            else events.fixedGridRhoFastjetAll
        )
        jets = self._add_jec_variables(jets, rho, isData)

        apply_jecs = bool(ak.any(jets.pt)) if (applyData or not isData) else False
        supported_nano = "v12" in nano_version or (
            year == "2024" and "v15" in nano_version
        )
        if not supported_nano:
            apply_jecs = False
        if not apply_jecs:
            return jets, None

        jec_vars = ["pt", "mass"]
        jet_factory_str = "ak8" if fatjets else "ak4"

        if (
            jet_factory_str not in self.jet_factory
            or self.jet_factory[jet_factory_str] is None
        ):
            print(f"No {jet_factory_str} JEC/JES factory available for {year}")
            return jets, None

        import cachetools

        if isData and year == "2024":
            jets_after_jec = self._build_2024_data_jets(
                events, jets, jet_factory_str
            )
        else:
            dataset = dataset or ""
            if isData:
                if year == "2022":
                    corr_key = "2022_runCD"
                elif year == "2022EE" and "Run2022E" in dataset:
                    corr_key = "2022EE_runE"
                elif year == "2022EE" and "Run2022F" in dataset:
                    corr_key = "2022EE_runF"
                elif year == "2022EE" and "Run2022G" in dataset:
                    corr_key = "2022EE_runG"
                elif year == "2023":
                    corr_key = "2023_runCv4" if "Run2023Cv4" in dataset else "2023_runCv123"
                elif year == "2023BPix":
                    corr_key = "2023BPix_runD"
                else:
                    print(
                        f"Warning: no data JEC factory configured for "
                        f"{dataset}, {year}"
                    )
                    return jets, None
            else:
                corr_key = f"{year}mc"

            if corr_key not in self.jet_factory[jet_factory_str]:
                raise KeyError(
                    f"No {jet_factory_str} JEC/JES factory key {corr_key!r}"
                )
            jets_after_jec = self.jet_factory[jet_factory_str][corr_key].build(
                jets, cachetools.Cache(np.inf)
            )

        # 2024 JER is absent from the pkl and is applied continuously from the
        # official JSON after JEC/JES. Data must never be JER-smeared.
        use_direct_jer = year == "2024" and not isData
        jer_shifted_vars = None
        if use_direct_jer:
            if "JER" in ak.fields(jets_after_jec):
                raise RuntimeError(
                    "The 2024 pkl already contains JER; remove its .jr/.jersf "
                    "inputs before enabling correctionlib JER"
                )
            jets, jer_shifted_vars = get_2024_jer(
                events, jets_after_jec, fatjets=fatjets
            )
        else:
            jets = jets_after_jec

        if jecs is None or isData:
            return jets, None

        # For 2024, apply the same nominal JER (and deterministic random draw)
        # to every JES-shifted collection. JER up/down are constructed directly
        # from the JSON SF uncertainty and added below.
        shifted_collections = {}
        for key, shift in jecs.items():
            if use_direct_jer and key == "JER":
                continue

            # The standalone total-uncertainty text file is exposed by this
            # pkl as JES_jes, while the existing skimmer requests JES. Keep the
            # public output name JES_up/down and resolve the internal alias
            # here. JES_Total is a fallback from UncertaintySources.
            shift_field = shift
            if shift_field not in ak.fields(jets_after_jec) and shift == "JES":
                shift_field = next(
                    (
                        candidate
                        for candidate in ("JES_jes", "JES_Total")
                        if candidate in ak.fields(jets_after_jec)
                    ),
                    None,
                )
            if shift_field is None or shift_field not in ak.fields(jets_after_jec):
                continue
            for direction in ("up", "down"):
                shifted_jets = jets_after_jec[shift_field][direction]
                if use_direct_jer:
                    shifted_jets, _ = get_2024_jer(
                        events,
                        shifted_jets,
                        fatjets=fatjets,
                        include_variations=False,
                    )
                shifted_collections[f"{key}_{direction}"] = shifted_jets

        jec_shifted_vars = {}
        for jec_var in jec_vars:
            variations = {"": jets[jec_var]}
            for label, shifted_jets in shifted_collections.items():
                variations[label] = shifted_jets[jec_var]
            if use_direct_jer and jer_shifted_vars is not None:
                variations["JER_up"] = jer_shifted_vars[jec_var]["JER_up"]
                variations["JER_down"] = jer_shifted_vars[jec_var]["JER_down"]
            jec_shifted_vars[jec_var] = variations

        return jets, jec_shifted_vars


def get_jmsr(
    fatjets: FatJetArray,
    num_jets: int,
    jmsr_vars: list[str],
    jms_values: dict,
    jmr_values: dict,
    isData: bool = False,
    seed: int = 42,
) -> dict:
    """Calculates post JMS/R masses and shifts"""

    jmsr_shifted_vars = {}

    for mkey in jmsr_vars:
        tdict = {}

        mass = pad_val(fatjets[mkey], num_jets, axis=1)
        jms = jms_values[mkey]
        jmr = jmr_values[mkey]

        if isData:
            tdict[""] = mass
        else:
            rng = np.random.default_rng(seed)
            smearing = rng.normal(size=mass.shape)
            # scale to JMR nom, down, up (minimum at 0)
            jmr_nom, jmr_down, jmr_up = ((smearing * max(jmr[i] - 1, 0) + 1) for i in range(3))
            jms_nom, jms_down, jms_up = jms

            corr_mass_JMRUp = random.gauss(0.0, jmr[2] - 1.0)
            corr_mass = max(jmr[0] - 1.0, 0.0) / (jmr[2] - 1.0) * corr_mass_JMRUp

            mass_jms = mass * jms_nom
            mass_jmr = mass * jmr_nom

            tdict[""] = mass * jms_nom * (1.0 + corr_mass)
            tdict["JMS_down"] = mass_jmr * jms_down
            tdict["JMS_up"] = mass_jmr * jms_up
            tdict["JMR_down"] = mass_jms * jmr_down
            tdict["JMR_up"] = mass_jms * jmr_up

        jmsr_shifted_vars[mkey] = tdict

    return jmsr_shifted_vars


# Jet Veto Maps
# the JERC group recommends ALL analyses use these maps, as the JECs are derived excluding these zones.
# apply to both Data and MC
# https://cms-talk.web.cern.ch/t/jet-veto-maps-for-run3-data/18444?u=anmalara
# https://cms-talk.web.cern.ch/t/jes-for-2022-re-reco-cde-and-prompt-fg/32873
def get_jetveto_event(jets: JetArray, year: str):
    """
    Get event selection that rejects events with jets in the veto map
    """

    # correction: Non-zero value for (eta, phi) indicates that the region is vetoed
    cset = correctionlib.CorrectionSet.from_file(get_pog_json_update("jetveto", year))
    j, nj = ak.flatten(jets), ak.num(jets)

    def get_veto(j, nj, csetstr):
        j_phi = np.clip(np.array(j.phi), -3.1415, 3.1415)
        j_eta = np.clip(np.array(j.eta), -4.7, 4.7)
        veto = cset[csetstr].evaluate("jetvetomap", j_eta, j_phi)
        return ak.unflatten(veto, nj)

    corr_str = {
        "2022": "Summer22_23Sep2023_RunCD_V1",
        "2022EE": "Summer22EE_23Sep2023_RunEFG_V1",
        "2023": "Summer23Prompt23_RunC_V1",
        "2023BPix": "Summer23BPixPrompt23_RunD_V1",
        "2024": "Summer24Prompt24_RunBCDEFGHI_V1",
    }[year]

    jet_veto = get_veto(j, nj, corr_str) > 0

    event_sel = ~(ak.any((jets.pt > 15) & jet_veto, axis=1))
    return event_sel
