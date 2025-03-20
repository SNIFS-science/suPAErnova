from typing import ClassVar, final

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.nflow import NFlowStepConfig


@final
class NFlowStep(SNPAEStep[NFlowStepConfig]):
    # Class Variables
    name: ClassVar["str"] = "nflow"


NFlowStep.register_step()
