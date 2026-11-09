from typing import Literal
from pydantic import BaseModel, Field
JsonPrimitive = Literal["string", "number", "boolean", "object", "array"]


class ParameterDefinition(BaseModel):
    """Defines a single parameter within a function's parameters dictionary.

    Attributes:
        arg_type: The expected JSON data type for this parameter, which is
            aliased to the 'type' field during serialization and validation.
    """
    arg_type: JsonPrimitive = Field(alias="type")


class ReturnDefinition(BaseModel):
    """Defines the expected return type for a function schema.

    Attributes:
        return_type: The data type returned by the function, which is aliased
            to the 'type' field during serialization and validation.
    """
    return_type: str = Field(alias="type")


class FunctionDefinition(BaseModel):
    """Represents a complete function schema available for the LLM to call.

    Attributes:
        name: The name of the function.
        description: A natural language description of the function's purpose.
        parameters: A mapping of parameter names to their respective type
            definitions. Defaults to an empty dictionary.
        returns: The definition of the function's return type.
    """
    name: str
    description: str
    parameters: dict[str, ParameterDefinition] = Field(default_factory=dict)
    returns: ReturnDefinition


class TestCase(BaseModel):
    """Represents a single input prompt for evaluation.

    Attributes:
        prompt: The natural language instruction or query provided by the user.
    """
    prompt: str
