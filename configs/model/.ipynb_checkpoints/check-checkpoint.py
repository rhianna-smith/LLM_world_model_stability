from litellm import completion

response = completion(
    model="bedrock/converse/google.gemma-3-4b-it",
    messages=[
        {"role": "user", "content": "hello"}
    ],
    max_tokens=20,
)

print(response)