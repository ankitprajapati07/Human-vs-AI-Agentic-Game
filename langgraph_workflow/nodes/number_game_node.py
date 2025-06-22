import logging
from langgraph_workflow.models import WorkflowState
from langchain_core.output_parsers import JsonOutputParser
from langgraph_workflow.prompt_templates.nodes_prompts import number_game_prompt
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import json

logger = logging.getLogger(__name__)

# Define the desired data structure.
class NumberGameModel(BaseModel):
    ai: str = Field(description="question asked by ai to user")
    status: str = Field(description="game status, either inprogress or done")

# Set up parser and instructions
parser = JsonOutputParser(pydantic_object=NumberGameModel)
format_instructions = parser.get_format_instructions()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

async def number_game_node(state: WorkflowState):
    """ Number Game Node for guessing the number """
    try:
        logger.info("\n################### NODE: number_game_node ###################")

        # Create the LangChain prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI running a number guessing game with the user."),
            ("user", f"{number_game_prompt}\n\n{format_instructions}")
        ])

        # Run prompt through LLM with parser
        chain = prompt | llm | parser
        llm_response = await chain.ainvoke({})

        # Ensure response is in dictionary format
        if isinstance(llm_response, NumberGameModel):
            llm_response = llm_response.dict()

        logger.debug(f"number_game_node - LLM response: {json.dumps(llm_response, indent=2)}")

        # Add the response to the state and return it
        updated_state = WorkflowState(
            **state.dict(),
            answer=llm_response.get("ai"),
            game_status=llm_response.get("status")
        )

        return updated_state

    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise e
