import os
import logging
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.tools import Tool
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
import requests

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GRAPHQL_API_URL = os.getenv("GRAPHQL_API_URL")

class GraphQLTool:
    def __init__(self, api_url: str):
        self.api_url = api_url
    
    def execute_query(self, query: str) -> Dict[str, Any]:
        logger.info(f"Executing GraphQL query: {query}")
        
        # Basic validation: Check for balanced braces and basic syntax
        if query.count('{') != query.count('}'):
            logger.error("Invalid GraphQL query: Unbalanced braces")
            raise ValueError("Invalid GraphQL query: Unbalanced braces")
        
        if not query.strip() or len(query.strip()) < 10:
            logger.error("Invalid GraphQL query: Query is empty or too short")
            raise ValueError("Invalid GraphQL query: Query is empty or too short")
        
        # Check for invalid characters in field names
        if any(c in query for c in '<>!@#$%^&*'):
            logger.error("Invalid GraphQL query: Contains invalid characters")
            raise ValueError("Invalid GraphQL query: Contains invalid characters")
        
        headers = {
            "Content-Type": "application/json",
        }
        
        payload = {
            "query": query
        }
        
        try:
            response = requests.post(self.api_url, headers=headers, json=payload)
            response.raise_for_status()
            result = response.json()
            logger.info(f"GraphQL response: {result}")
            return result
        except requests.exceptions.RequestException as e:
            logger.error(f"GraphQL query failed: {str(e)}")
            raise

graphql_tool = GraphQLTool(GRAPHQL_API_URL)

def create_agent():
    llm = AzureChatOpenAI(
        azure_endpoint=os.getenv("OPENAI_API_ENDPOINT"),
        api_version=os.getenv("OPENAI_API_VERSION"),
        api_key=os.getenv("OPENAI_API_CREDENTIAL"),
        temperature=0
    )
    
    tools = [
        Tool(
            name="GraphQLTool",
            func=graphql_tool.execute_query,
            description="""
            Use this tool to execute GraphQL queries against the Jobs API.
            The input should be a valid GraphQL query string.
            The tool will return the JSON response from the GraphQL API.
            """
        )
    ]
    
    system_message = """
    You are a helpful assistant that translates natural language queries into valid GraphQL queries for a Jobs API.

    Your task is to:
    1. Understand the user's question about jobs.
    2. Translate it into a syntactically correct GraphQL query.
    3. Execute the query using the GraphQLTool.
    4. Interpret the results and provide a human-readable response.

    The GraphQL API schema includes:
    - Query: `jobs(filter: JobFilter, sort: JobSort, limit: Int): [Job]`
    - Type Job: `{ id: ID, title: String, description: String, location: String, salary: Int }`
    - Input JobFilter: `{ location: String }`
    - Input JobSort: `{ field: String, order: String }`

    Rules for creating GraphQL queries:
    - Use proper GraphQL syntax with correct braces and field names.
    - For filtering by location, use `filter: { location: "value" }`.
    - For sorting, use `sort: { field: "salary", order: "DESC" }` for descending order.
    - Limit results with `limit: 10` for top results.
    - Only include fields: `id`, `title`, `description`, `location`, `salary`.
    - Avoid extra braces, invalid field names, or incorrect syntax.
    - Ensure the query is enclosed in a single `{}` block.

    Example query for top-paying jobs in London:
    ```graphql
    {
      jobs(filter: { location: "London" }, sort: { field: "salary", order: "DESC" }, limit: 10) {
        title
        location
        salary
      }
    }
    ```

    If the query is ambiguous, make reasonable assumptions (e.g., sort by salary for "top paying").
    Always validate the query syntax before execution and provide clear, concise responses based on the data.
    If the API returns an error, interpret it and suggest a corrected query if possible.
    """
    
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_message),
            HumanMessagePromptTemplate.from_template("{input}")
        ]
    )
    
    agent = create_openai_tools_agent(llm, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True
    )
    
    return agent_executor

async def process_query(query: str) -> str:
    agent_executor = create_agent()
    result = await agent_executor.ainvoke({"inputbat": query})
    return result["output"]