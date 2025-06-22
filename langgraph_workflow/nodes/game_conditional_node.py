import logging
from langgraph_workflow.models import WorkflowState
from langgraph_workflow.config import games_dict

logger = logging.getLogger(__name__)

def route_to_game(state: WorkflowState):
    """ Route to the game based on the current game type.
        Task: Implement the logic to route to the appropriate game.
    """
    try:
        # Retrieve the current game type from the state
        current_game = state.current_game_type

        # Validate if the current game type exists in games_dict
        if current_game not in games_dict:
            raise ValueError(f"Invalid game type: {current_game}")

        # Log the selected game type
        logger.info(f"Routing to game: {current_game}")

        # Return the appropriate game based on current_game_type
        return games_dict[current_game]

    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise e
