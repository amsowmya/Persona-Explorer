import asyncio 
from typing import cast, Any, Literal 
import json 
import os
from dotenv import load_dotenv

from tavily import AsyncTavilyClient
from langchain_openai import AzureChatOpenAI
from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from pydantic import BaseModel, Field 

from src.agent.configuration import Configuration
from src.agent.state import InputState, OutputState, OverallState
from src.agent.utils import deduplicate_and_format_sources, format_all_notes

from src.agent.prompts import (
    QUERY_WRITER_PROMPT,
    INFO_PROMPT,
    EXTRACTION_PROMPT,
    REFLECTION_PROMPT
)

load_dotenv()

os.environ['AZURE_OPENAI_API_KEY'] = os.getenv('AZURE_OPENAI_API_KEY')


llm = AzureChatOpenAI(
    api_version="2024-08-01-preview",
    azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
    azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
)

tavily_search_client = AsyncTavilyClient()


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries."
    )


class ReflectionOutput(BaseModel):
    is_satisfactory: bool = Field(
        description="True if all required fields are well populated, False otherwise"
    )
    missing_fields: list[str] = Field(
        description="List of fields names that are missing or incomplete"
    )
    search_queries: list[str] = Field(
        description="If is_satisfactory is False, provide 1-3 targeted search queries to find the missing information"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")



def generate_queries(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    configurable = Configuration.from_runnable_config(config)
    max_search_queries = configurable.max_search_queries

    # Generate search queries 
    structured_llm = llm.with_structured_output(Queries, method="function_calling")

    # Format system instructions
    person_str = f"Email: {state.person['email']}"
    if "name" in state.person:
        person_str += f" Name: {state.person['name']}"
    if "linkedin" in state.person:
        person_str += f" Linkedin URL: {state.person['linkedin']}"
    if "role" in state.person:
        person_str += f" Role: {state.person['role']}"
    if "company" in state.person:
        person_str += f" Company: {state.person['company']}"

    query_instructions = QUERY_WRITER_PROMPT.format(
        person=person_str,
        info=json.dumps(state.extraction_schema, indent=2),
        user_notes=state.user_notes,
        max_search_queries=max_search_queries
    )

    results = cast(
        Queries,
        structured_llm.invoke(
            [
                {"role": "system", "content": query_instructions},
                {"role": "user", 
                 "content": "Please generate a list of search queries related to the schema that you want to populate."},
            ]
        ),
    )

    query_list = [query for query in results.queries]
    return {"search_queries": query_list}

async def research_person(state: OverallState, config: RunnableConfig) -> dict[str, Any]:
    """Execute a multi-step web search and information extraction process.
    
    This function performs the following steps:
    1. Executes concurrent web searches using the Tavily API.
    2. Deduplicates and formats the search results.
    """

    configurable = Configuration.from_runnable_config(config)
    max_search_results = configurable.max_search_results

    # Web search 
    search_tasks = []

    for query in state.search_queries:
        search_tasks.append(
            tavily_search_client.search(
                query,
                days=360,
                max_results=max_search_results,
                include_raw_content=True,
                topic="general",
            )
        )

    # Execute all searches concurrently
    # all the searches are executed at the same time, rather than one after the other, which improves performance.
    search_docs = await asyncio.gather(*search_tasks)

    # Deduplicate and format sources 
    source_str = deduplicate_and_format_sources(
        search_docs, max_tokens_per_source=1000, include_raw_content=True
    )

    # Generate structured notes relevant to the extraction schema
    p = INFO_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2),
        content=source_str,
        people=state.person,
        user_notes=state.user_notes
    )

    result = await llm.ainvoke(p)
    return {"completed_notes": [str(result.content)]}

def gather_notes_extract_schema(state: OverallState) -> dict[str, Any]:
    """Gather notes from the web search and extract the schema fields."""

    # Format all notes 
    notes = format_all_notes(state.completed_notes)

    # Extract schema fields
    system_prompt = EXTRACTION_PROMPT.format(
        info=json.dumps(state.extraction_schema, indent=2), notes=notes
    )

    structured_llm = llm.with_structured_output(state.extraction_schema, method="function_calling")
    result = structured_llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", 
             "content": "Produce a structured output from these notes."
            },
        ]
    )
    return {"info": result}


def reflection(state: OverallState) -> dict[str, Any]:
    """Reflect on the extracted information and generate search queries to find missing information."""
    structured_llm = llm.with_structured_output(ReflectionOutput, method="function_calling")

    # Format reflection prompt
    system_prompt = REFLECTION_PROMPT.format(
        schema=json.dumps(state.extraction_schema, indent=2),
        info=state.info
    )

    result = cast(
        ReflectionOutput,
        structured_llm.invoke(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "Produce a structured reflection output."}
            ]
        ),
    )

    if result.is_satisfactory:
        return {"is_satisfactory": result.is_satisfactory}
    else:
        return {
            "is_satisfactory": result.is_satisfactory,
            "search_queries": result.search_queries,
            "reflection_steps_taken": state.reflection_steps_taken + 1,
        }

def route_from_reflection(
        state: OverallState, config: RunnableConfig
) -> Literal[END, "research_person"]:
    """Route the graph based on the reflection output."""
    configurable = Configuration.from_runnable_config(config)

    if state.is_satisfactory:
        return END 
    
    if state.reflection_steps_taken <= configurable.max_reflection_steps:
        return "research_person"
    
    return END


builder = StateGraph(
    OverallState,
    input=InputState,
    output=OutputState,
    config_schema=Configuration
)

builder.add_node("generate_queries", generate_queries)
builder.add_node("research_person", research_person)
builder.add_node("gather_notes_extract_schema", gather_notes_extract_schema)
builder.add_node("reflection", reflection)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "research_person")
builder.add_edge("research_person", "gather_notes_extract_schema")
builder.add_edge("gather_notes_extract_schema", "reflection")
builder.add_conditional_edges("reflection", route_from_reflection)

builder.add_edge("research_person", END)    


graph = builder.compile()


# graph.invoke({"person": [("user", "Email: soumya.pdit@gmail.com")]})

# Define an async function to invoke the graph
async def main():
    result = await graph.ainvoke({
        "person": {
            "email": "sowmya.am@gmail.com",
            "name": "Sowmya AM"
        }
    })
    print("==== RESULT ====")
    print(result)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())