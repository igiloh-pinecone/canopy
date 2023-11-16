from typing import Optional, Any

from openai import AzureOpenAI

from canopy.llm import OpenAILLM


class AzureOpenAILLM(OpenAILLM):
    """
    Azure OpenAI LLM wrapper built on top of the OpenAI Python client.

    Note: Azure OpenAI services requires a valid Azure API key and a valid Azure endpoint to use this class.
          You can set the "AZURE_OPENAI_KEY" and "AZURE_OPENAI_ENDPOINT" environment variables to your API key and
          endpoint, respectively, or you can directly, e.g.:
          >>> from openai import AzureOpenAI
          >>> AzureOpenAI.api_key = "YOUR_AZURE_API_KEY"
          >>> AzureOpenAI.api_base = "YOUR_AZURE_ENDPOINT"

    Note: If you want to pass an OpenAI organization, you need to set an environment variable "OPENAI_ORG_ID". Note
          that this is different from the environment variable name for passing an organization to the parent class,
          OpenAILLM, which is "OPENAI_ORG".

          You cannot currently set this environment variable manually, as shown above.
    """
    def __init__(self,
                 *,
                 azure_api_version: str,
                 model_name: str,
                 azure_api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 **kwargs: Any
                 ):
        """
        Initialize the AzureOpenAI LLM.

        Args:
            azure_api_version: The Aure OpenAI API version, e.g. "2023-05-15".
                      Find versions here: https://learn.microsoft.com/en-us/rest/api/azureopenai/files
            model_name: The name of the deployed Azure model you are connecting to. This is *not* the name of the
                      OpenAI LLM underlying your deployed Azure model.
            azure_api_key: Azure API key. Find at https://portal.azure.com >> Resource Management.
            base_url: Azure endpoint. Find at https://portal.azure.com >> Resource Management.
            **kwargs: Generation default parameters to use for each request. See https://platform.openai.com/docs/api-reference/chat/create
                      For example, you can set the temperature, top_p, etc.
                      These params can be overridden by passing a `model_params` argument to the `chat_completion` or
                      `enforced_function_call` methods.
        """
        super().__init__(model_name)
        self.azure_api_version = azure_api_version
        self.model_name = model_name
        self.azure_api_key = azure_api_key
        self.base_url = base_url

        self._client = AzureOpenAI(azure_endpoint=self.base_url,
                                   api_key=self.azure_api_key,
                                   api_version=self.azure_api_version
                                   )

        self.default_model_params = kwargs


if __name__ == "__main__":
    AZURE_OPENAI_ENDPOINT = "https://devrel.openai.azure.com/"
    AZURE_OPENAI_KEY = "d5e870aabbb14cb09a52038215c0fb37"
    llm = AzureOpenAILLM(azure_api_version="2023-07-01-preview",
                         azure_api_key=AZURE_OPENAI_KEY,
                         base_url=AZURE_OPENAI_ENDPOINT,
                         model_name="audrey_canopy_test")

    print('hi')