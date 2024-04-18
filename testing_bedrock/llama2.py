import os
import json
import sys
import boto3

print("impotred successfully...")

prompt="""

        you are a cricket expert now just tell me when RCB will win the IPL?
"""

bedrock=boto3.client(service_name="bedrock-runtime")

payload={
    
    "prompt": "[INST]"+prompt+"[/INST]",
    "max_gen_len": 512,
    "temperature": 0.3,
    "top_p":0.9
}

body=json.dumps(payload)
model_id="meta.llama2-70b-chat-v1"


response=bedrock.invoke_model(
    body=body,
    modelId=model_id,
    accept="application/json",
    contentType="application/json"
)

response_body=json.loads(response.get("body").read())
response_text=response_body["generation"]
print(response_text)