from pydantic import BaseModel, Field

positive_int = Field(gt=0, description="Must be a positive integer")

class InputData(BaseModel):
    """
    Input data that can be added to the data
    """
    
class InputDataPredict(BaseModel):
    """
    Input data for the prediction of the consommation
    """