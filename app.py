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
requests_get = TextRequestsWrapper()
