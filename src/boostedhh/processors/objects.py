"""Common object definitions."""

from __future__ import annotations

import awkward as ak
import numpy as np


def jetid_v12(jets: ak.Array) -> ak.Array:
    """
    Jet ID fix for NanoAOD v12 copying
    # https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/117#note_8880716
    """

    jetidtightbit = (jets.jetId & 2) == 2
    jetidtight = (
        ((np.abs(jets.eta) <= 2.7) & jetidtightbit)
        | (
            ((np.abs(jets.eta) > 2.7) & (np.abs(jets.eta) <= 3.0))
            & jetidtightbit
            & (jets.neHEF < 0.99)
        )
        | ((np.abs(jets.eta) > 3.0) & jetidtightbit & (jets.neEmEF < 0.4))
    )

    jetidtightlepveto = (
        (np.abs(jets.eta) <= 2.7) & jetidtight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8)
    ) | ((np.abs(jets.eta) > 2.7) & jetidtight)

    return jetidtight, jetidtightlepveto


def jetid_v14(jets: ak.Array) -> tuple[ak.Array, ak.Array]:
    """
    Jet ID fix for NanoAOD v14 copying
    # https://gitlab.cern.ch/cms-jetmet/coordination/coordination/-/issues/117#note_8880788
    """

    jetidtight = (
        (
            (np.abs(jets.eta) <= 2.6)
            & (jets.neHEF < 0.99)
            & (jets.neEmEF < 0.9)
            & ((jets.chMultiplicity + jets.neMultiplicity) > 1)
            & (jets.chHEF > 0.01)
            & (jets.chMultiplicity > 0)
        )
        | (
            ((np.abs(jets.eta) > 2.6) & (np.abs(jets.eta) <= 2.7))
            & (jets.neHEF < 0.90)
            & (jets.neEmEF < 0.99)
        )
        | (((np.abs(jets.eta) > 2.7) & (np.abs(jets.eta) <= 3.0)) & (jets.neHEF < 0.99))
        | ((np.abs(jets.eta) > 3.0) & (jets.neMultiplicity >= 2) & (jets.neEmEF < 0.4))
    )

    jetidtightlepveto = (
        (np.abs(jets.eta) <= 2.7) & jetidtight & (jets.muEF < 0.8) & (jets.chEmEF < 0.8)
    ) | ((np.abs(jets.eta) > 2.7) & jetidtight)

    return jetidtight, jetidtightlepveto
