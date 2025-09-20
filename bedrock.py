import boto3
import json
from botocore.exceptions import ClientError

client = boto3.client("bedrock-runtime", region_name="ap-south-1")

# Use your inference profile ARN (as you already have)
model_id = "arn:aws:bedrock:ap-south-1:217522444118:inference-profile/apac.amazon.nova-pro-v1:0"

user_message = "Write a poem for Coding"

body = {
    "messages": [
        {
            "role": "user",
            "content": [{"text": user_message}],
        }
    ],
    "inferenceConfig": {
        "maxTokens": 512,
        "temperature": 0.5,
        "topP": 0.9
    }
}

try:
    response = client.invoke_model(
        modelId=model_id,
        body=json.dumps(body),          # ✅ messages go here
        contentType="application/json",
        accept="application/json"
    )

    response_body = json.loads(response["body"].read())
    response_text = response_body["output"]["message"]["content"][0]["text"]

    print(response_text)

except (ClientError, Exception) as e:
    print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
    exit(1)
