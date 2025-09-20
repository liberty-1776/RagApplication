import boto3
import json
import nltk

prompt_data="\n\nHuman: Act as a Shakespeare and write a poem on Genertaive AI\n\nAssistant:"

bedrock=boto3.client(service_name="bedrock-runtime")


body = {
    "anthropic_version": "bedrock-2023-05-31",
    "messages": [
        {"role": "user", "content": "Act as Shakespeare and write a poem on Generative AI"}
    ],
    "max_tokens": 512,
    "temperature": 0.8,
    "top_p": 0.8
}

model_id = "anthropic.claude-3-sonnet-20240229-v1:0"
response = bedrock.invoke_model(
    body=json.dumps(body),
    modelId=model_id,
    accept="application/json",
    contentType="application/json",
)


response_body = json.loads(response.get("body").read())

print(response_body["content"][0]["text"])