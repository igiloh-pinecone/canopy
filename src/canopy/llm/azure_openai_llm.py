import os
from typing import Optional, Any

from openai import AzureOpenAI

from canopy.llm import OpenAILLM


class AzureOpenAILLM(OpenAILLM):
    """
    Azure OpenAI LLM wrapper built on top of the OpenAI Python client.

    Note: Azure OpenAI services requires an Azure API key, Azure endpoint, and OpenAI API version.

    a valid Azure API key and a valid Azure endpoint to use this class.
          You can set the "AZURE_OPENAI_KEY" and "AZURE_OPENAI_ENDPOINT" environment variables to your API key and
          endpoint, respectively, or you can directly, e.g.:
          >>> from openai import AzureOpenAI
          >>> AzureOpenAI.api_key = "YOUR_AZURE_API_KEY"
          >>> AzureOpenAI.api_base = "YOUR_AZURE_ENDPOINT"
          >>> AzureOpenAI.api_version = "THE_AZURE_API_VERSION_YOU_ARE_USING"

    Note: If you want to pass an OpenAI organization, you need to set an environment variable "OPENAI_ORG_ID". Note
          that this is different from the environment variable name for passing an organization to the parent class,
          OpenAILLM, which is "OPENAI_ORG".

          You cannot currently set this environment variable manually, as shown above.
    """
    def __init__(self,
                 model_name: str = "gpt-3.5-turbo",  # why do i need this?
                 *,
                 api_key:  Optional[str] = None,
                 azure_deployment: Optional[str] = None,
                 azure_endpoint: Optional[str] = None,
                 azure_api_version: Optional[str] = None,
                 azure_api_key: Optional[str] = None,
                 **kwargs: Any,
                 ):
        """
        Initialize the AzureOpenAI LLM.

        >>> os.environ['OPENAI_API_VERSION'] = "AZURE API VERSION"
        >>> os.environ['AZURE_OPENAI_ENDPOINT'] = "AZURE ENDPOINT"
        >>> os.environ['AZURE_OPENAI_API_KEY'] = "YOUR KEY"
        >>> os.environ['AZURE_DEPLOYMENT'] = "YOUR AZURE DEPLOYMENT'S NAME"

        >>> from canopy.models.data_models import UserMessage
        >>> llm = AzureOpenAILLM()
        >>> messages = [UserMessage(content="Hello! How are you?")]
        >>> llm.chat_completion(messages)

        """
        super().__init__(model_name)

        if azure_deployment is None:
            if os.getenv('AZURE_DEPLOYMENT') is None:
                raise EnvironmentError('You need to set an environment variable for AZURE_DEPLOYMENT to the name of '
                                       'your Azure deployment')
            azure_deployment = os.getenv('AZURE_DEPLOYMENT')

        self._client = AzureOpenAI(azure_deployment=azure_deployment,
                                   azure_endpoint=azure_endpoint,
                                   api_version=azure_api_version,
                                   api_key=azure_api_key,
                                   )

        self.default_model_params = kwargs

