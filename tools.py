from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

def save_to_txt(data:str, filename:str = "research.txt"):
    timetamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"---Research Output---\n{timetamp}\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)

        return f"Data saved to {filename}."

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="A tool to save the research output to a text file.",
    
)

search = DuckDuckGoSearchRun
search_tool = Tool(
    name="search",
    func=search.run,
    description="A tool to search the web for information. Use this tool to find information on a specific topic.",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)