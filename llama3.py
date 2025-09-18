import boto3
import json

prompt_data = "Act as Shakespeare and write a poem on Generative AI"

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    "prompt":prompt_data,
    "max_gen_len":512,
    "temperature":0.5,
    "top_p":0.9
}
body=json.dumps(payload)
model_id="meta.llama3-70b-instruct-v1:0"
response = bedrock.invoke_model(
    body=json.dumps(payload),
    modelId="meta.llama3-70b-instruct-v1:0",
    accept="application/json",
    contentType="application/json"
)


# Parse body
response_body = json.loads(response.get("body").read())

# For LLaMA models in Bedrock, the key is usually `generation`
print(response_body)

if "generation" in response_body:
    print("Generated text:", response_body["generation"])
elif "outputs" in response_body:
    print("Generated text:", response_body["outputs"][0]["text"])
else:
    print("No text found in response")