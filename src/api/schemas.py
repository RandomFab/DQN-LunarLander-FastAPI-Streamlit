from pydantic import BaseModel, Field, field_validator
from typing import List


class Observation(BaseModel):
    """
    Schema for the LunarLander-v3 observation vector.
    The state is an 8-dimensional vector: [x, y, vx, vy, angle, v_angle, left_leg, right_leg]
    """

    state: List[float] = Field(
        ...,
        min_length=8,
        max_length=8,
        description="LunarLander state vector of exactly 8 floating point numbers.",
        json_schema_extra={
            "example": [0.12, 1.45, 0.0, -0.1, 0.05, 0.0, 0, 0]
        }
    )

    @field_validator("state")
    @classmethod
    def validate_state_values(cls, v: List[float]) -> List[float]:
        # You can add specific range checks here if needed,
        # or just confirm they are all finite numbers.
        if any(not (isinstance(x, (int, float))) for x in v):
            raise ValueError("All elements in the state must be numbers.")
        return v


class PredictionResponse(BaseModel):
    """
    Schema for the model's prediction output.
    """

    action: int = Field(..., description="Predicted action (0: None, 1: Left, 2: Main, 3: Right)")

    model_config = {
        "json_schema_extra": {
            "view_example": {"action": 2}
        }
    }
