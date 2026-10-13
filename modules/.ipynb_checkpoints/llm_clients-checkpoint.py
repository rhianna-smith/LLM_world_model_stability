import json
import threading
import time
from typing import Optional, TypeVar

import litellm
from litellm import completion
from pydantic import BaseModel, ValidationError

from modules.schemas import LLMAnswer

SchemaT = TypeVar("SchemaT", bound=BaseModel)

litellm.num_retries = 0


class LiteLLMCompletionClient:
    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        max_retries: int,
        seed: Optional[int] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.seed = seed

    def call(
        self,
        prompt: str,
        instruction_template: str,
        response_schema: type[SchemaT] = LLMAnswer,
    ) -> SchemaT:
        thread_id = threading.get_ident()

        # print(f"[Thread {thread_id}] Entered call()")

        retryable_errors = (
            litellm.APIConnectionError,
            litellm.APIError,
            litellm.RateLimitError,
            litellm.ServiceUnavailableError,
            litellm.Timeout,
        )

        is_openai_model = self.model_name.startswith("openai/")
        is_bedrock_model = "bedrock" in self.model_name.lower()

        # print(
        #     f"[Thread {thread_id}] "
        #     f"is_openai_model={is_openai_model}, "
        #     f"is_bedrock_model={is_bedrock_model}"
        # )

        schema_json = response_schema.model_json_schema()

        if not is_openai_model:
            # print(f"[Thread {thread_id}] Building JSON schema prompt")

            instruction_template = f"""
{instruction_template}

Return ONLY a valid JSON object.

Do not include:
- markdown
- explanations
- prose
- code fences

The response must parse with json.loads().

Wrap the JSON in <json></json> tags.

Example:
<json>
{{"key": "value"}}
</json>

JSON schema:
{json.dumps(schema_json, indent=2)}
"""

            # print(f"[Thread {thread_id}] JSON schema prompt built")

        for attempt in range(self.max_retries):
            # print(
            #     f"[Thread {thread_id}] "
            #     f"Starting attempt {attempt + 1}/{self.max_retries}"
            # )

            try:
                # print(f"[Thread {thread_id}] Building kwargs")

                kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {
                            "role": "system",
                            "content": instruction_template,
                        },
                        {
                            "role": "user",
                            "content": prompt,
                        },
                    ],
                    "temperature": self.temperature,
                    "top_p": 0.9,
                    "max_tokens": self.max_tokens,
                    "timeout": 60,
                }

                # print(f"[Thread {thread_id}] Base kwargs built")

                if is_openai_model:
                    # print(f"[Thread {thread_id}] Adding OpenAI response_format")

                    kwargs["response_format"] = response_schema

                is_gemma_model = "gemma" in self.model_name.lower()

                if is_bedrock_model and not is_gemma_model:
                    # print(f"[Thread {thread_id}] Adding Bedrock stop sequences")

                    kwargs["stop"] = [
                        "</json>",
                        "```",
                    ]

                if (
                    self.seed is not None
                    and not is_bedrock_model
                ):
                    # print(f"[Thread {thread_id}] Adding seed")

                    kwargs["seed"] = self.seed

                # print(
                    # f"[Thread {thread_id}] "
                    # f"Calling completion() with model={self.model_name}"
                # )

                start_time = time.time()

                response = completion(**kwargs)

                elapsed = time.time() - start_time

                # print(
                #     f"[Thread {thread_id}] "
                #     f"completion() returned in {elapsed:.2f}s"
                # )

                # print(f"[Thread {thread_id}] Extracting content")

                content = response.choices[0].message.content

                if isinstance(content, list):
                    content = "".join(
                        block.text
                        for block in content
                        if hasattr(block, "text")
                    )

                content = content.strip()

                # print(
                #     f"[Thread {thread_id}] "
                #     f"Raw content length={len(content)}"
                # )

                json_start = content.find("<json>")

                if json_start != -1:
                    # print(f"[Thread {thread_id}] Extracting JSON from <json> tag")

                    content = content[json_start + len("<json>"):].strip()

                json_end = content.find("</json>")

                if json_end != -1:
                    # print(f"[Thread {thread_id}] Removing </json> tag")

                    content = content[:json_end].strip()

                if content.startswith("```json"):
                    # print(f"[Thread {thread_id}] Removing ```json fences")

                    content = (
                        content.removeprefix("```json")
                        .removesuffix("```")
                        .strip()
                    )

                elif content.startswith("```"):
                    # print(f"[Thread {thread_id}] Removing ``` fences")

                    content = (
                        content.removeprefix("```")
                        .removesuffix("```")
                        .strip()
                    )

                # print(
                #     f"[Thread {thread_id}] "
                #     f"Validating JSON into {response_schema.__name__}"
                # )

                validated = response_schema.model_validate_json(content)

                # print(f"[Thread {thread_id}] Validation succeeded")

                return validated

            except retryable_errors as error:
                wait_time = 2**attempt

                print(
                    f"[Thread {thread_id}] "
                    f"Retryable LiteLLM error on attempt "
                    f"{attempt + 1}/{self.max_retries}: "
                    f"{type(error).__name__}: {error}"
                )

                print(
                    f"[Thread {thread_id}] "
                    f"Sleeping for {wait_time}s before retry"
                )

                time.sleep(wait_time)

            except (json.JSONDecodeError, ValidationError) as error:
                wait_time = 2**attempt

                print(
                    f"[Thread {thread_id}] "
                    f"Validation error on attempt "
                    f"{attempt + 1}/{self.max_retries}: "
                    f"{type(error).__name__}: {error}"
                )

                print(
                    f"[Thread {thread_id}] "
                    f"Sleeping for {wait_time}s before retry"
                )

                time.sleep(wait_time)

            except Exception as error:
                # print(
                #     f"[Thread {thread_id}] "
                #     f"NON-RETRYABLE ERROR: "
                #     f"{type(error).__name__}: {error}"
                # )

                raise

        # print(f"[Thread {thread_id}] Max retries exceeded")

        raise RuntimeError("Max retries exceeded")