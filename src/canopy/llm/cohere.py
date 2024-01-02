import time
from copy import deepcopy
from typing import Union, Iterable, Optional, Any, Dict, List

import cohere
from tenacity import retry, stop_after_attempt

from canopy.llm import BaseLLM
from canopy.llm.models import Function
from canopy.models.api_models import (
    _StreamChoice,
    ChatResponse,
    StreamingChatChunk,
    TokenCounts,
)
from canopy.models.data_models import Context, Messages, Role, Query


class CohereLLM(BaseLLM):
    """
    Cohere LLM wrapper built on top of the Cohere Python client.

    Note: Cohere requires a valid API key to use this class.
          You can set the "CO_API_KEY" environment variable to your API key.
    """
    def __init__(self,
                 model_name: Optional[str] = "command-nightly",
                 *,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any,
                 ):
        """
        Initialize the Cohere LLM.

        Args:
            model_name: The name of the model to use. See https://docs.cohere.com/docs/models
            api_key: Your Cohere API key. Defaults to None (uses the "CO_API_KEY" environment variable).
            base_url: The base URL to use for the Cohere API. Defaults to None (uses the "CO_API_URL" environment variable if set, otherwise use default Cohere API URL).
            **kwargs: Generation default parameters to use for each request. See https://platform.openai.com/docs/api-reference/chat/create
                    For example, you can set the temperature, p, etc
                    These params can be overridden by passing a `model_params` argument to the `chat_completion` methods.
        """  # noqa: E501
        super().__init__(model_name)
        self._client = cohere.Client(api_key, api_url=base_url)
        self.default_model_params = kwargs

    def chat_completion(self,
                        system_prompt: str,
                        chat_history: Messages,
                        context: Optional[Context] = None,
                        *,
                        stream: bool = False,
                        max_tokens: Optional[int] = None,
                        model_params: Optional[dict] = None,
                        ) -> Union[ChatResponse, Iterable[StreamingChatChunk]]:
        """
        Chat completion using the Cohere API.

        Note: this function is wrapped in a retry decorator to handle transient errors.

        Args:
            system_prompt: The system prompt to use for the chat completion (preamble).
            chat_history: Messages (chat history) to send to the model.
            context: Knowledge base context to use for the chat completion. Defaults to None (no context).
            stream: Whether to stream the response or not.
            max_tokens: Maximum number of tokens to generate. Defaults to None (generates until stop sequence or until hitting max context size).
            model_params: Model parameters to use for this request. Defaults to None (uses the default model parameters).
                          Dictonary of parametrs to override the default model parameters if set on initialization.
                          For example, you can pass: {"temperature": 0.9, "top_p": 1.0} to override the default temperature and top_p.
                          see: https://platform.openai.com/docs/api-reference/chat/create
        Returns:
            ChatResponse or StreamingChatChunk

        Usage:
            >>> from canopy.llm import OpenAILLM
            >>> from canopy.models.data_models import UserMessage
            >>> llm = CohereLLM()
            >>> messages = [UserMessage(content="Hello! How are you?")]
            >>> result = llm.chat_completion(messages)
            >>> print(result.choices[0].message.content)
            "I'm good, how are you?"
        """  # noqa: E501
        model_params_dict: Dict[str, Any] = deepcopy(self.default_model_params)
        model_params_dict.update(
            model_params or {}
        )
        connectors = model_params_dict.pop('connectors', None)
        messages: Dict[str: Any] = self.map_messages(chat_history)

        if not messages:
            raise cohere.error.CohereAPIError("No message provided")

        response = self._client.chat(
            model=self.model_name,
            message=messages[-1]['message'],
            chat_history=messages[:-1],
            preamble_override=system_prompt,
            stream=stream,
            connectors=[
                {"id": connector} for connector in connectors
            ] if connectors else None,
            max_tokens=max_tokens,
            **model_params_dict
        )

        def streaming_iterator(res):
            for chunk in res:
                if chunk.event_type != "text-generation":
                    continue

                choice = _StreamChoice(
                    index=0,
                    delta={
                        "content": chunk.text,
                        "function_call": None,
                        "role": Role.ASSISTANT,
                        "tool_calls": None
                    },
                    finish_reason=None,
                )
                streaming_chat_chunk = StreamingChatChunk(
                    id='',
                    object="chat.completion.chunk",
                    created=int(time.time()),
                    model=self.model_name,
                    choices=[choice],
                )
                streaming_chat_chunk.id = chunk.id

                yield streaming_chat_chunk

        if stream:
            return streaming_iterator(response)

        return ChatResponse(
            id=response.id,
            created=int(time.time()),
            choices=[{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response.text,
                },
                "finish_reason": "stop",
            }],
            object="chat.completion",
            model=self.model_name,
            usage=TokenCounts(
                prompt_tokens=response.token_count["prompt_tokens"],
                completion_tokens=response.token_count["response_tokens"],
                total_tokens=response.token_count["billed_tokens"],
            ),
        )

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
    )
    def generate_search_queries(self, messages):
        messages = self.map_messages(messages)
        response = self._client.chat(
            model=self.model_name,
            message=messages[-1]['message'],
            chat_history=messages[:-1],
            stream=False,
            search_queries_only=True,
        )
        return [search_query['text'] for search_query in response.search_queries]

    def enforced_function_call(self,
                               messages: Messages,
                               function: Function,
                               *,
                               max_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None,) -> dict:
        return NotImplementedError()

    async def aenforced_function_call(self,
                                      system_prompt: str,
                                      chat_history: Messages,
                                      function: Function, *,
                                      max_tokens: Optional[int] = None,
                                      model_params: Optional[dict] = None):
        raise NotImplementedError()

    async def achat_completion(self,
                               messages: Messages, *, stream: bool = False,
                               max_generated_tokens: Optional[int] = None,
                               model_params: Optional[dict] = None,
                               ) -> Union[ChatResponse,
                                          Iterable[StreamingChatChunk]]:
        raise NotImplementedError()

    async def agenerate_queries(self,
                                messages: Messages,
                                *,
                                max_generated_tokens: Optional[int] = None,
                                model_params: Optional[dict] = None,
                                ) -> List[Query]:
        raise NotImplementedError()

    def map_messages(self, messages: Messages) -> List[dict]:
        """
        Map the messages to format expected by Cohere.

        Cohere Chat API expects message history to be in the format:
        {
          "role": "USER",
          "message": "message text"
        }

        System messages will be passed as user messages.

        Args:
            messages: (chat history) to send to the model.

        Returns:
            list A List of dicts in format expected by Cohere chat API.
        """
        mapped_messages = []

        for message in messages:
            if not message.content:
                continue

            mapped_messages.append({
                "role": "CHATBOT" if message.role == Role.ASSISTANT else "USER",
                "message": message.content,
            })

        return mapped_messages
