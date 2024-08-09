# https://ubiops.com/docs/r_client_library/deployment_requests/#response-structure_1
from qcog_python_client.schema.generated_schema.models import TrainingStatus

WAITING_STATUS = (TrainingStatus.processing, TrainingStatus.pending)
SUCCESS_STATUS = (TrainingStatus.completed,)
