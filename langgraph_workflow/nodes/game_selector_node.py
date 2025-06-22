import logging
from langgraph_workflow.models import WorkflowState
from utils import load_session, load_required_game_data
from langchain_openai import OpenAI

# Use OpenAI LLM using LangChain
llm = OpenAI(model="gpt-4o", temperature=0)

logger = logging.getLogger(__name__)


async def game_selector_node(state: WorkflowState):
    """ Select the game type based on user input.
        Task: Implement the logic to select and validate the game type.
    """
    try:
        logger.info("\n################### NODE: game_selector_node ###################")

        # Load the session data by retrieving 'session id' and 'session file' from the state
        session_id = state.session_id
        session_file = state.session_file
        game_data = load_session(session_id, session_file)

        # Load chat history from game data
        chat_history = load_required_game_data(game_data)
        logger.debug(f"chat_history: {chat_history}")

        # Extract game type (you can replace this with logic using `llm` if needed)
        current_game_type = game_data.get("game_type", None)
        if not current_game_type:
            raise ValueError("Game type not found in session data.")

        # Add game type and chat history to the state and return it
        updated_state = WorkflowState(
            **state.dict(),  # copy existing state
            current_game_type=current_game_type,
            chat_history=chat_history
        )
        return updated_state

    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise e
