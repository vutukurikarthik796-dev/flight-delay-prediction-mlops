from typing import List
from pydantic import BaseModel, Field, field_validator


class FlightInput(BaseModel):
    """Flight details known before departure; scheduled_departure uses HHMM (e.g. 1830)."""
    MONTH: int = Field(ge=1, le=12, description="Month number, 1-12")
    DAY: int = Field(ge=1, le=31, description="Day of month, 1-31")
    DAY_OF_WEEK: int = Field(ge=1, le=7, description="Day of week, 1-7")
    AIRLINE: str = Field(min_length=1, max_length=10, description="Airline code, e.g. AA")
    ORIGIN_AIRPORT: str = Field(min_length=1, max_length=10, description="Origin airport code, e.g. JFK")
    DESTINATION_AIRPORT: str = Field(min_length=1, max_length=10, description="Destination airport code, e.g. LAX")
    SCHEDULED_TIME: float = Field(gt=0, le=2000, description="Scheduled flight duration in minutes")
    DISTANCE: float = Field(ge=0, le=20000, description="Flight distance")
    scheduled_departure: int = Field(ge=0, le=2359, description="Scheduled departure time in HHMM, e.g. 1830")

    @field_validator("scheduled_departure")
    @classmethod
    def validate_hhmm(cls, value: int) -> int:
        hour, minute = divmod(value, 100)
        if hour > 23 or minute > 59:
            raise ValueError("scheduled_departure must be a valid HHMM time, e.g. 1830")
        return value

    @field_validator("AIRLINE", "ORIGIN_AIRPORT", "DESTINATION_AIRPORT")
    @classmethod
    def normalize_code(cls, value: str) -> str:
        value = value.strip().upper()
        if not value:
            raise ValueError("Code must not be blank")
        return value


class PredictionResponse(BaseModel):
    prediction: int = Field(description="0 = not delayed over 15 minutes; 1 = delayed over 15 minutes")
    prediction_label: str
    delay_probability: float = Field(ge=0.0, le=1.0)


class BatchPredictionRequest(BaseModel):
    flights: List[FlightInput] = Field(min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    count: int
    predictions: List[PredictionResponse]
