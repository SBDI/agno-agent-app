from enum import Enum
from typing import List, Optional, Tuple

from agents.sage import get_sage
from agents.scholar import get_scholar


class AgentType(Enum):
    SAGE = "sage"
    SCHOLAR = "scholar"


class ModelProvider(Enum):
    OPENAI = "openai"
    GROQ = "groq"


def get_available_agents() -> List[str]:
    """Returns a list of all available agent IDs."""
    return [agent.value for agent in AgentType]


def get_model_provider(model_id: str) -> ModelProvider:
    """Determines the provider for a given model ID.

    Args:
        model_id: The ID of the model to check

    Returns:
        The provider of the model (OPENAI or GROQ)
    """
    if model_id.startswith(("llama", "mixtral")):
        return ModelProvider.GROQ
    else:
        return ModelProvider.OPENAI


def get_available_models() -> List[Tuple[str, ModelProvider]]:
    """Returns a list of all available models with their providers.

    Returns:
        A list of tuples containing (model_id, provider)
    """
    return [
        ("gpt-4o", ModelProvider.OPENAI),
        ("o3-mini", ModelProvider.OPENAI),
        ("llama-3.3-70b-versatile", ModelProvider.GROQ),
        ("llama-3.3-8b-versatile", ModelProvider.GROQ),
        ("mixtral-8x7b-32768", ModelProvider.GROQ),
    ]


def get_agent(
    model_id: str = "gpt-4o",
    agent_id: Optional[AgentType] = None,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    debug_mode: bool = True,
):
    """Creates and returns an agent of the specified type with the specified model.

    Args:
        model_id: The ID of the model to use
        agent_id: The type of agent to create
        user_id: The ID of the user
        session_id: The ID of the session
        debug_mode: Whether to enable debug mode

    Returns:
        An agent of the specified type with the specified model
    """
    # Determine which agent to create
    if agent_id == AgentType.SAGE:
        return get_sage(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
    else:
        return get_scholar(model_id=model_id, user_id=user_id, session_id=session_id, debug_mode=debug_mode)
