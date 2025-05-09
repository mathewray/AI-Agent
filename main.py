from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool,save_tool
import os

# Load environment variables from .env file
load_dotenv()
print("Loaded API Key:", os.getenv("OPENAI_API_KEY"))

class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]



#Initialize the LLM and the output parser
llm = ChatOpenAI(model="gpt-4o-mini")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

# Define the prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """

            You are a student at a university. You are tasked with researching a topic and writing an essay about it.
            You will be provided with a topic and a list of sources. Your task is to summarize the topic, provide a list of sources, and describe the tools you used to research the topic.
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
      ("placeholder","{chat_history}"),
      ("human","{query}"),
      ("placeholder","{agent_scratchpad}"),  
    ]
).partial(format_instructions=parser.get_format_instructions())

#Define the tools to be used by the agent
tools = [search_tool, wiki_tool,save_tool]
agent = create_tool_calling_agent(
    llm=llm,
    prompt=prompt,
    tools=tools

)

# Create the agent executor
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
# Get user input and invoke the agent executor
query = input("What can I help you research? ")
raw_response = agent_executor.invoke(
    {
        "query":query
        #"tool_input": query #Pass the query as the tool input 
    }
)
print(raw_response)

# Parse response
try:
    structured_response = parser.parse(raw_response.get("output")[0]["text"])
    print(structured_response)
except Exception as e:
    print("Error parsing response:",e, "Raw Response - ",raw_response)