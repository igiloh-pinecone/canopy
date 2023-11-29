from typing import List, Optional

from canopy.chat_engine.models import HistoryPruningMethod
from canopy.chat_engine.prompt_builder import PromptBuilder
from canopy.chat_engine.query_generator import QueryGenerator
from canopy.llm import BaseLLM, CohereLLM
from canopy.models.data_models import Messages, Query


class CohereQueryGenerator(QueryGenerator):
    """
    Query generator for LLM clients that have a built-in feature to
    generate search queries from chat messages.
    """
    _DEFAULT_COMPONENTS = {
        "llm": CohereLLM,
    }

    def __init__(self,
                 *,
                 llm: Optional[BaseLLM] = None):
        self._llm = llm or self._DEFAULT_COMPONENTS["llm"]()
        self._prompt_builder = PromptBuilder(HistoryPruningMethod.RAISE, 1)

    def generate(self,
                 messages: Messages,
                 max_prompt_tokens: int) -> List[Query]:
        queries = self._llm.generate_search_queries(messages)
        return [Query(text=query) for query in queries]

    async def agenerate(self,
                        messages: Messages,
                        max_prompt_tokens: int) -> List[Query]:
        raise NotImplementedError
