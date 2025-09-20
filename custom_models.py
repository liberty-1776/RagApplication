from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import List
from pydantic import PrivateAttr
import boto3, json

class DeepSeekChatModel(BaseChatModel):
    model_id: str = "deepseek.v3-v1:0"
    region: str = "ap-south-1"
    max_tokens: int = 512
    temperature: float = 0.7

    # ✅ Declare boto3 client as a private attribute
    _client: any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        # Convert LangChain messages → Bedrock schema
        body = {
            "messages": [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant",
                 "content": m.content}
                for m in messages
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response["body"].read())

        # Extract text safely depending on schema
        text = response_body["choices"][0]["message"]["content"]

        ai_message = AIMessage(content=text)
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
        )


    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
    
class ChatGptModel(BaseChatModel):
    model_id: str = "openai.gpt-oss-120b-1:0"
    region: str = "ap-south-1"
    max_tokens: int = 512
    temperature: float = 0.7

    # ✅ Declare boto3 client as a private attribute
    _client: any = PrivateAttr()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._client = boto3.client("bedrock-runtime", region_name=self.region)

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs):
        # Convert LangChain messages → Bedrock schema
        body = {
            "messages": [
                {"role": "user" if isinstance(m, HumanMessage) else "assistant",
                 "content": m.content}
                for m in messages
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = self._client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json",
        )
        response_body = json.loads(response["body"].read())

        # Extract text safely depending on schema
        text = response_body["choices"][0]["message"]["content"]

        ai_message = AIMessage(content=text)
        return ChatResult(
            generations=[ChatGeneration(message=ai_message)],
        )


    @property
    def _llm_type(self) -> str:
        return "deepseek-chat"
