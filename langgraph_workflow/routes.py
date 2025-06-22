import logging
from fastapi import APIRouter, Depends
from utils import (
    create_response,
    initialize_session,
    update_state_result,
    process_user_input_update_session_data
)
import os
from langgraph_workflow.models import ChatRequest
from langgraph_workflow.workflow import LanggraphAgents
import uuid

# Create a directory to store session files
SESSION_DIR = "sessions_data"
os.makedirs(SESSION_DIR, exist_ok=True)

router = APIRouter(prefix="/agent", tags=["agent"])

lang_agent = LanggraphAgents()
logger = logging.getLogger(__name__)


session_id = str(uuid.uuid4())


@router.post("/game")
async def chat(request: ChatRequest, graph=Depends(lang_agent.build_workflow)):
    try:
        logger.info("========= Inside Game API =========")

        global session_id
        logger.debug(f"Request Session ID: '{session_id}'")
        session_file = os.path.join(SESSION_DIR, f"{session_id}.json")

        # Step 1: Initialize the session
        initialize_session(session_id, session_file)

        # Step 2: Process user input and update session
        data = process_user_input_update_session_data(session_file, request)

        if request.user_input == "result":
            return create_response(message="Results fetched successfully", status_code=200, success=True, data=data)

        ############### Start the workflow ###############

        # TODO: Initialize state with user input
        initial_state = {
            "session_id": session_id,
            "session_file": session_file,
            "user_input": request.user_input  # include actual user input
        }

        # Step 3: Invoke the LangGraph workflow
        result_state = await graph.ainvoke(input=initial_state)

        # Step 4: Update result data in session or database
        update_state_result(result_state)

        # TODO: Add appropriate response to data_res dict from result_state
        data_res = {
            "game_type": result_state.get("current_game_type"),
            "answer": result_state.get("answer"),
            "next_step": result_state.get("next_step", None),  # optional
        }

        logger.debug(f"Answer for the user query: '{result_state.get('answer')}'")
        return create_response(message="Answer retrieved successfully", status_code=200, success=True, data=data_res)

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return create_response(message="Something went wrong while retrieving the answer", status_code=500,
                               success=False, data={})
