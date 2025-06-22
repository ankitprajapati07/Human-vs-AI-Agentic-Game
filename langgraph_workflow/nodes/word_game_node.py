import logging
from langgraph_workflow.models import WorkflowState
from langchain_core.output_parsers import JsonOutputParser
from langgraph_workflow.prompt_templates.nodes_prompts import word_game_prompt
from langgraph_workflow.config import word_game_list
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

logger = logging.getLogger(__name__)

# Define your desired data structure
class NumberGameModel(BaseModel):
    ai: str = Field(description="question asked by ai to user")
    status: str = Field(description="game status, either inprogress or done")

# Set up a parser + inject instructions into the prompt template
parser = JsonOutputParser(pydantic_object=NumberGameModel)
format_instructions = parser.get_format_instructions()

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

async def word_game_node(state: WorkflowState):
    """ Word Game Node for guessing the word """
    try:
        logger.info("\n################### NODE: word_game_node ###################")

        # Create the prompt using LangChain ChatPromptTemplate
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are playing a word guessing game with the user."),
            ("user", f"{word_game_prompt}\n\n{format_instructions}")
        ])

        # Call the LLM using LangChain's chain method
        chain = prompt | llm | parser
        llm_response = await chain.ainvoke({})

        # Validate structure and convert to dict if needed
        if isinstance(llm_response, NumberGameModel):
            llm_response = llm_response.dict()

        # Log the response
        logger.debug(f"word_game_node - LLM response: {json.dumps(llm_response, indent=2)}")

        # Add the LLM response to state and return it
        updated_state = WorkflowState(
            **state.dict(),
            answer=llm_response.get("ai"),
            game_status=llm_response.get("status")
        )

        return updated_state

    except Exception as e:
        logger.error(f"Exception: {str(e)}")
        raise e
