from typing import Literal, Any
from pydantic import BaseModel, Field


class ParameterDefinition(BaseModel):
    """Defines a single parameter for a callable function."""
    arg_type: Literal["string", "number", "boolean", "object", "array"]


class ReturnDefinition(BaseModel):
    """Defines the return type schema."""
    return_type: str


class FunctionDefinition(BaseModel):
    """Defines the schema of an available function for the LLM."""
    name: str = Field(pattern=r"^fn_")
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ReturnDefinition


class OutputRecord(BaseModel):
    """Represents the final structured function call extracted from the LLM."""
    prompt: str
    name: str = Field(pattern=r"^fn_")
    parameters: dict[str, Any] = Field(default_factory=dict)
