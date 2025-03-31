from typing import ClassVar, final

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.nflow import NFlowStepConfig


class NFlowStep(SNPAEStep[NFlowStepConfig]):
    # Class Variables
    id: ClassVar["str"] = "nflow"


NFlowStep.register_step()
