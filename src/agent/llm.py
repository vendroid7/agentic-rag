"""Groq LLM wrapper — text and structured JSON output."""

import json

from pydantic import ValidationError
from groq import Groq


class LLMClient:
    """
    Client interface for interacting with the Groq LLM API.

    Handles routing between different model tiers (small vs large) and 
    enforces structural constraints on outputs via JSON mode and Pydantic validation.
    """

    def __init__(self, api_key: str, small_model: str, large_model: str, temperature: float = 0.2, max_tokens: int = 2048):
        self.client = Groq(api_key=api_key)
        self.small = small_model
        self.large = large_model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _model(self, size):
        return self.small if size == "small" else self.large

    def text(self, instructions: str, question: str, model: str = "small") -> str:
        """
        Generates an unstructured natural language response.

        Args:
            instructions (str): The system prompt defining the persona and rules.
            question (str): The user query alongside formatted retrieval context.
            model (str): The model tier to utilize ('small' or 'large'). Defaults to 'small'.

        Returns:
            str: The raw text response synthesized by the LLM.
        """
        response = self.client.chat.completions.create(
            model=self._model(model),
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()

    def structured(self, instructions: str, question: str, schema: type, model: str = "small"):
        """
        Generates a strictly typed JSON response validated against a Pydantic schema.

        Leverages the provider's JSON mode alongside schema injection in the prompt
        to guarantee structural compliance for downstream state processing.

        Args:
            instructions (str): The system prompt defining the logic.
            question (str): The input payload or query to process.
            schema (type): The Pydantic model class to validate the output against.
            model (str): The model tier to utilize. Defaults to 'small'.

        Returns:
            BaseModel: An instantiated Pydantic object of the specified schema type.

        Raises:
            ValueError: If the generated JSON fails to validate against the schema.
        """
        full_instructions = (
            f"{instructions}\n\n"
            "Reply with JSON only, no prose, matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        response = self.client.chat.completions.create(
            model=self._model(model),
            messages=[
                {"role": "system", "content": full_instructions},
                {"role": "user", "content": question},
            ],
            temperature=self.temperature,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        reply = response.choices[0].message.content.strip()
        try:
            return schema.model_validate_json(reply)
        except ValidationError as e:
            raise ValueError(f"LLM reply doesn't match {schema.__name__}: {reply}") from e
