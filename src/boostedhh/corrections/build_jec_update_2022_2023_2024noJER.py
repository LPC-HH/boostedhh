from __future__ import annotations

import contextlib
from pathlib import Path

from coffea.jetmet_tools import CorrectedJetsFactory, CorrectedMETFactory, JECStack
from coffea.lookup_tools import extractor

jec_name_map = {
    "JetPt": "pt",
    "JetMass": "mass",
    "JetEta": "eta",
    "JetA": "area",
    "ptGenJet": "pt_gen",
    "ptRaw": "pt_raw",
    "massRaw": "mass_raw",
    "Rho": "event_rho",
    "METpt": "pt",
    "METphi": "phi",
    "JetPhi": "phi",
    "UnClusteredEnergyDeltaX": "MetUnclustEnUpDeltaX",
    "UnClusteredEnergyDeltaY": "MetUnclustEnUpDeltaY",
}


def jet_factory_factory(files):
    ext = extractor()
    with contextlib.ExitStack() as stack:
        real_files = [stack.enter_context(Path(f"data/jecs_jin_819/{f}")) for f in files]
        ext.add_weight_sets([f"* * {file}" for file in real_files])
        ext.finalize()

    jec_stack = JECStack(ext.make_evaluator())
    return CorrectedJetsFactory(jec_name_map, jec_stack)


jet_factory = {
    # MC: JEC/JES plus legacy JER for 2022--2023BPix.
    "2022mc": jet_factory_factory(
        files=[
            "Summer22_22Sep2023_V4_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023_V4_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023_V4_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer22_22Sep2023_V4_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Summer2222Sep2023_JRV2_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Summer2222Sep2023_JRV2_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022EEmc": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023_V4_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023_V4_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023_V4_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer22EE_22Sep2023_V4_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Summer22EE22Sep2023_JRV2_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Summer22EE22Sep2023_JRV2_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    "2023mc": jet_factory_factory(
        files=[
            "Summer23Prompt23_V4_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V4_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V4_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer23Prompt23_V4_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Summer23Prompt23RunCv1234_JRV3_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Summer23Prompt23RunCv1234_JRV3_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    "2023BPixmc": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_V4_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V4_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V4_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer23BPixPrompt23_V4_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
            "Summer23BPixPrompt23RunD_JRV3_MC_PtResolution_AK4PFPuppi.jr.txt.gz",
            "Summer23BPixPrompt23RunD_JRV3_MC_SF_AK4PFPuppi.jersf.txt.gz",
        ],
    ),
    # 2024 JER is handled separately with correctionlib JSON.
    "2024mc": jet_factory_factory(
        files=[
            "Summer24Prompt24_V5_MC_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_V5_MC_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_V5_MC_UncertaintySources_AK4PFPuppi.junc.txt.gz",
            "Summer24Prompt24_V5_MC_Uncertainty_AK4PFPuppi.junc.txt.gz",
        ],
    ),
    # Data: L1FastJet, L2Relative and the applicable residual payload.
    "2022_runCD": jet_factory_factory(
        files=[
            "Summer22_22Sep2023run355065_V4_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023run355065_V4_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023run355065_V4_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runE": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023run359022_V4_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run359022_V4_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run359022_V4_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runF": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023run360332_V4_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run360332_V4_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run360332_V4_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runG": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023run362350_V4_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run362350_V4_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run362350_V4_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv123": jet_factory_factory(
        files=[
            "Summer23Prompt23_run366365_V4_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run366365_V4_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run366365_V4_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv4": jet_factory_factory(
        files=[
            "Summer23Prompt23_run367765_V4_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run367765_V4_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run367765_V4_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2023BPix_runD": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_run369803_V4_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_run369803_V4_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_run369803_V4_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run378971": jet_factory_factory(
        files=[
            "Summer24Prompt24_run378971_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run378971_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run378971_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run380253": jet_factory_factory(
        files=[
            "Summer24Prompt24_run380253_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380253_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380253_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run380948": jet_factory_factory(
        files=[
            "Summer24Prompt24_run380948_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380948_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380948_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run381944": jet_factory_factory(
        files=[
            "Summer24Prompt24_run381944_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run381944_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run381944_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run382298": jet_factory_factory(
        files=[
            "Summer24Prompt24_run382298_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run382298_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run382298_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run383247": jet_factory_factory(
        files=[
            "Summer24Prompt24_run383247_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383247_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383247_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run383780": jet_factory_factory(
        files=[
            "Summer24Prompt24_run383780_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383780_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383780_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run384933": jet_factory_factory(
        files=[
            "Summer24Prompt24_run384933_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run384933_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run384933_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run385814": jet_factory_factory(
        files=[
            "Summer24Prompt24_run385814_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run385814_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run385814_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run386409": jet_factory_factory(
        files=[
            "Summer24Prompt24_run386409_V5_DATA_L1FastJet_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run386409_V5_DATA_L2Relative_AK4PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run386409_V5_DATA_L2L3Residual_AK4PFPuppi.jec.txt.gz",
        ],
    ),
}

fatjet_factory = {
    # MC: JEC/JES plus legacy JER for 2022--2023BPix.
    "2022mc": jet_factory_factory(
        files=[
            "Summer22_22Sep2023_V4_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023_V4_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023_V4_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer22_22Sep2023_V4_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Summer2222Sep2023_JRV2_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Summer2222Sep2023_JRV2_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ],
    ),
    "2022EEmc": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023_V4_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023_V4_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023_V4_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer22EE_22Sep2023_V4_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Summer22EE22Sep2023_JRV2_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Summer22EE22Sep2023_JRV2_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ],
    ),
    "2023mc": jet_factory_factory(
        files=[
            "Summer23Prompt23_V4_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V4_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_V4_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer23Prompt23_V4_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Summer23Prompt23RunCv1234_JRV3_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Summer23Prompt23RunCv1234_JRV3_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ],
    ),
    "2023BPixmc": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_V4_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V4_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_V4_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer23BPixPrompt23_V4_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
            "Summer23BPixPrompt23RunD_JRV3_MC_PtResolution_AK8PFPuppi.jr.txt.gz",
            "Summer23BPixPrompt23RunD_JRV3_MC_SF_AK8PFPuppi.jersf.txt.gz",
        ],
    ),
    # 2024 JER is handled separately with correctionlib JSON.
    "2024mc": jet_factory_factory(
        files=[
            "Summer24Prompt24_V5_MC_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_V5_MC_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_V5_MC_UncertaintySources_AK8PFPuppi.junc.txt.gz",
            "Summer24Prompt24_V5_MC_Uncertainty_AK8PFPuppi.junc.txt.gz",
        ],
    ),
    # Data: L1FastJet, L2Relative and the applicable residual payload.
    "2022_runCD": jet_factory_factory(
        files=[
            "Summer22_22Sep2023run355065_V4_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023run355065_V4_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22_22Sep2023run355065_V4_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runE": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023run359022_V4_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run359022_V4_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run359022_V4_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runF": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023run360332_V4_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run360332_V4_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run360332_V4_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2022EE_runG": jet_factory_factory(
        files=[
            "Summer22EE_22Sep2023run362350_V4_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run362350_V4_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer22EE_22Sep2023run362350_V4_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv123": jet_factory_factory(
        files=[
            "Summer23Prompt23_run366365_V4_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run366365_V4_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run366365_V4_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2023_runCv4": jet_factory_factory(
        files=[
            "Summer23Prompt23_run367765_V4_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run367765_V4_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23Prompt23_run367765_V4_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2023BPix_runD": jet_factory_factory(
        files=[
            "Summer23BPixPrompt23_run369803_V4_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_run369803_V4_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer23BPixPrompt23_run369803_V4_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run378971": jet_factory_factory(
        files=[
            "Summer24Prompt24_run378971_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run378971_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run378971_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run380253": jet_factory_factory(
        files=[
            "Summer24Prompt24_run380253_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380253_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380253_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run380948": jet_factory_factory(
        files=[
            "Summer24Prompt24_run380948_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380948_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run380948_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run381944": jet_factory_factory(
        files=[
            "Summer24Prompt24_run381944_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run381944_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run381944_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run382298": jet_factory_factory(
        files=[
            "Summer24Prompt24_run382298_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run382298_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run382298_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run383247": jet_factory_factory(
        files=[
            "Summer24Prompt24_run383247_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383247_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383247_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run383780": jet_factory_factory(
        files=[
            "Summer24Prompt24_run383780_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383780_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run383780_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run384933": jet_factory_factory(
        files=[
            "Summer24Prompt24_run384933_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run384933_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run384933_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run385814": jet_factory_factory(
        files=[
            "Summer24Prompt24_run385814_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run385814_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run385814_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
    "2024_run386409": jet_factory_factory(
        files=[
            "Summer24Prompt24_run386409_V5_DATA_L1FastJet_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run386409_V5_DATA_L2Relative_AK8PFPuppi.jec.txt.gz",
            "Summer24Prompt24_run386409_V5_DATA_L2L3Residual_AK8PFPuppi.jec.txt.gz",
        ],
    ),
}

met_factory = CorrectedMETFactory(jec_name_map)


if __name__ == "__main__":
    import argparse
    import gzip

    import cloudpickle

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--output", default="jec_compiled_update.pkl.gz", type=str)
    args = parser.parse_args()

    with gzip.open(args.output, "wb") as fout:
        cloudpickle.dump(
            {
                "jet_factory": jet_factory,
                "fatjet_factory": fatjet_factory,
                "met_factory": met_factory,
            },
            fout,
        )
