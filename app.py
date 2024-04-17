import os
from crewai import Agent, Task, Crew
from crewai_tools import SerperDevTool
from langchain_openai import ChatOpenAI
from langchain.tools import tool
import requests
#from unstructured.partition.html import partition_html
import json
from bs4 import BeautifulSoup
from textwrap import dedent
from pydantic import ValidationError
from typing import List
from langchain_openai import OpenAI
from langchain.agents import initialize_agent, load_tools
from langchain_community.document_loaders import AsyncChromiumLoader
from langchain_community.document_transformers import BeautifulSoupTransformer
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate
from crewai_tools import WebsiteSearchTool
from langchain_community.utilities import TextRequestsWrapper
from fastapi import FastAPI, HTTPException, Request
requests_get = TextRequestsWrapper()
OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)
#TOOLS
# To enable the tool to search any website the agent comes across or learns about during its operation
website_tool = WebsiteSearchTool()

@tool
def browser_tool_with_soup(url) -> str:
    """Useful to scrape and summarize a website content, just pass a string with
    only the full url, no need for a final slash `/`, eg: https://google.com or https://clearbit.com/about-us"""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/56.0.2924.76 Safari/537.36', "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"}
    #response = requests.request("POST", url, headers=headers, data=payload)
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    content = BeautifulSoup(response.content, 'html.parser')
    content=" ".join(content.stripped_strings)
    #content = [content[i:i + 8000] for i in range(0, len(content), 8000)]
    summaries = []
    return f'\nScrapped Content: {content}\n'

class SearchTools():

  @tool("Search internet")
  def search_internet(query):
    """Useful to search the internet about a given topic and return relevant
    results."""
    return SearchTools.search(query)

  @tool("Search opentable")
  def search_opentable(query):
    """Useful to search for opetable to find resturants and their availability and booking information."""
    query = f"site:opentable.com {query}"
    return SearchTools.search(query)

  def search(query, n_results=5):
    url = "https://google.serper.dev/search"
    payload = json.dumps({"q": query})
    headers = {
        'X-API-KEY': os.environ['SERPER_API_KEY'],
        'content-type': 'application/json'
    }
    response = requests.request("POST", url, headers=headers, data=payload)
    results = response.json()['organic']
    stirng = []
    for result in results[:n_results]:
      try:
        stirng.append('\n'.join([
            f"Title: {result['title']}", f"Link: {result['link']}",
            f"Snippet: {result['snippet']}", "\n-----------------"
        ]))
      except KeyError:
        next

    content = '\n'.join(stirng)
    return f"\nSearch result: {content}\n"

#AGENTS
finder_agent = Agent(
    role='Venue Researcher',
    goal='Find at least 10 venues in the provided LOCATION. find places with great reviews that have not permanently closed. Gather enough information so that your coworker can use the information for other tasks.Do not use websites that do not allow scraping such as tripadvisor',
    backstory='A knowledgeable local guide with extensive information about the city and its neighbourhoods, its resturants, bars, venues, parks, events and attractions',
    tools=[
					SearchTools.search_internet,
         website_tool,
        SearchTools.search_opentable
			],
			verbose=False,
      llm=OpenAIGPT35
)


writer_agent = Agent(
    role='Writer',
    goal=dedent(f""""\Synthesize the data collected by your coworkers to present to the user.
    Format the output in the following format.
             Name:
          Address:
          Reviews:
          Recent Reviews:
          Phone Number:
          Booking Process:
          """),
    backstory='You are experienced in synthesing data and writing catchy specific descriptions',
    verbose=False,
    llm=OpenAIGPT35
)

#TASKS
class BookingTasks:
    def find_venues_task(agent, user_need):
        return Task(
            description=dedent(f"""\
      Search and discover at least 10 results for user query:  {user_need}.
      You should identify places with great reviews and have not permanently closed. Gather each identified venues Name, address, reviews, recent reviews, phone number and address. Avoid sites like tripadvisor that do not allow scraping.The year is 2024.
			"""),
            expected_output=dedent(f"""
                For at least 10 places, find the following information:
                Name:
                Address:
                Reviews:
                Recent Reviews:
                Phone Number:
                Booking Process:
            """),
            agent=agent
        )

def writer(agent):
    return Task(
        description=dedent(f"""
            You will consolidate all the data and format it in an easily understandable form.
        """),
        expected_output=dedent(f"""\
          Print out what your coworkers have found.
          Follow this output template
          Name:
          Address:
          Reviews:
          Recent Reviews:
          Phone Number:
          Booking Process:

          Example of output
          Manna Korean Restaurant
          Location: Manly, Sydney
          Rating: 4.5 stars
          Review: "Amazing food and great service. Highly recommended!"
          Status: Currently open
      """),
        agent=agent
    )

def find_venue(user_need):
  task1 = BookingTasks.find_venues_task(agent=finder_agent,user_need=user_need)
  task2 = writer(agent=writer_agent)
  # Initialize the crew with tasks
  crew = Crew(
      agents=[finder_agent, writer_agent],
      tasks=[task1, task2],
      verbose=False
  )
  result = crew.kickoff()
  return result

app = FastAPI()
@app.post("/venue_finder")
async def venue_finder(request: Request):
    form_data = await request.form()
    input_prompt = form_data.get('user_prompt')
    user_id = form_data.get('user_id')
    task1 = BookingTasks.find_venues_task(agent=finder_agent, user_need=input_prompt)

    task2 = writer(agent=writer_agent)
    # Initialize the crew with tasks
    crew = Crew(
        agents=[finder_agent, writer_agent],
        tasks=[task1, task2],
        verbose=False
    )
    
    # Execute the crew
    result = crew.kickoff()
    return {"text": result, "unique_id": user_id}
