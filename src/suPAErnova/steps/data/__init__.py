from typing import ClassVar, final

from suPAErnova.steps import SNPAEStep
from suPAErnova.configs.steps.data import DataStepConfig


@final
class DataStep(SNPAEStep[DataStepConfig]):
    # Class Variables
    name: ClassVar["str"] = "data"


DataStep.register_step()
