from typing import ClassVar, final

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.posterior import PosteriorStepConfig


@final
class PosteriorStep(SNPAEStep[PosteriorStepConfig]):
    # Class Variables
    name: ClassVar["str"] = "nflow"


PosteriorStep.register_step()
