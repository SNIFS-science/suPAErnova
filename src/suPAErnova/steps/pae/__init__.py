from typing import ClassVar, final

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.pae import PAEStepConfig


@final
class PAEStep(SNPAEStep[PAEStepConfig]):
    # Class Variables
    name: ClassVar["str"] = "pae"


PAEStep.register_step()
