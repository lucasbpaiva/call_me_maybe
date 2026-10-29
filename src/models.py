from typing import Literal, Any
from pydantic import BaseModel, Field
JsonPrimitive = Literal["string", "number", "boolean", "object", "array"]


class ParameterDefinition(BaseModel):
    """Defines a single parameter for a callable function."""
    arg_type: JsonPrimitive = Field(alias="type")


class ReturnDefinition(BaseModel):
    """Defines the return type schema."""
    return_type: str = Field(alias="type")


class FunctionDefinition(BaseModel):
    """Defines the schema of an available function for the LLM."""
    name: str = Field(pattern=r"^fn_")
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ReturnDefinition


class TestCase(BaseModel):
    """Defines a single test case from the input JSON."""
    prompt: str


class OutputRecord(BaseModel):
    """Represents the final structured function call extracted from the LLM."""
    prompt: str
    name: str = Field(pattern=r"^fn_")
    parameters: dict[str, Any] = Field(default_factory=dict)
