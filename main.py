from typing import Optional
from google import genai
import asyncio
from google.genai.client import Client
from google.genai.types import Any, GenerateContentResponse
from google.generativeai.types.generation_types import GenerationConfigDict
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import bs4
import time

load_dotenv()

class PromptSchema(BaseModel):
    """
    Schema for the prompt the orchestrator gives to the worker LLMs.
    section_name: Name of the section, as HTML ID
    prompt: Brief Description of the section
    """
    section_name: str
    prompt: str

class PlanningResponse(BaseModel):
    """
    Response Model for Planning out the website
    theme_context: Theming colours, values, padding values, etc.
    shared_context: Context which will be shared between LLMs when generating website.
    prompts: List of prompts which will be supplied to another LLM to generate output of the website. Should be individual sections of a page.
    skeleton: The framework of the website, no actual code.
    """
    theme_context: str
    shared_context: str
    prompts: list[PromptSchema]
    skeleton: str

class WorkerResponse(BaseModel):
    """
    Response Model for the workers creating sections of the website.
    html_code: The HTML code for the section the worker is generating
    css_code: Any optional CSS. Do not include any tags, only CSS
    js_code: Any optional Javascript. Do not include any tags, only Javascript
    """
    html_code: str
    css_code: Optional[str]
    js_code: Optional[str]

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
worker_clients: list[Client] = [genai.Client(api_key=os.getenv("GEMINI_API_KEY_"+str(i%3))) for i in range(10)]

async def generate_section(prompt: str, shared_context: str, theme_context: str, section_name: str, index: int = 0):
    worker = worker_clients[index % len(worker_clients)]
    worker_response = await worker.aio.models.generate_content(
        model="gemini-2.0-flash-lite",
        contents=f"""
        This is context about the website you are building: {shared_context}
        This is the theming of the website: {theme_context}
        Generate HTML Code for this prompt: {prompt}
        You are a worker being orchestrated by a master LLM.
        You have been assigned only this section.
        Only generate this section, nothing else.
        Use Tailwind CSS for styling.
        For images, use placeholder.png.
        Make the design look modern and futuristic.
        Include Custom JS and CSS for that section if needed.
        Add interactivity in the elements if needed.
        Also include any custom font.
        """,
        config={
            "response_mime_type": "application/json",
            "response_schema": WorkerResponse,
        }
    )
    return (section_name, worker_response.parsed.html_code, worker_response.parsed.css_code, worker_response.parsed.js_code)

async def main():
    prompt = input(" > ")
    plan_response: GenerateContentResponse = await client.aio.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"""
        This is the prompt of the user: {prompt}
        The user wants to create a landing page.
        The landing page should be as big and useful as it can be.
        Create a plan of action which multiple LLMs will follow to build the website.
        The plan of action should be the different sections on the landing page.
        The plan of action must contain prompts which will be given to the website generation model.
        Also, supply the HTML code containing the basic structure of the website, including the sections with their ID as the section name.
        DO NOT ADD ANY CODE EXCEPT BOILERPLATE/SKELETON CODE.
        Make sure you set the margins and padding to the body correctly.
        The prompt should be detailed.
        Use Tailwind for styling.
        This is the tag for TailwindCSS: <script src="https://unpkg.com/@tailwindcss/browser@4"></script>
        Include the Tailwind import tag in the skeleton.
        Put all repetitive information into the shared context.
        Add website style, colours, font theming, font colours, etc in the theme context.
        For images, use placeholder.png.
        Set a font if needed.
        Share key details like font, colour scheme, sizing values and ratios in the context.
        """,
        config={
            "response_mime_type": "application/json",
            "response_schema": PlanningResponse,
        }
    )
    print("Theme:", plan_response.parsed.theme_context)
    skeleton_soup = bs4.BeautifulSoup(plan_response.parsed.skeleton, "html.parser")
    tasks = []
    for i, plan_item in enumerate(plan_response.parsed.prompts):
        tasks.append(generate_section(plan_item.prompt, plan_response.parsed.shared_context, plan_response.parsed.theme_context, plan_item.section_name, i))
    results = await asyncio.gather(*tasks)
    css = []
    js = []
    for section, html_snippet, css_snippet, js_snippet in results:
        css.append(css_snippet)
        js.append(js_snippet)
        skeleton_soup.find(id=section).replace_with(html_snippet)
    css = "\n".join([x for x in css if x is not None])
    js = "\n".join([x for x in js if x is not None])
    skeleton_soup.head.insert(1, f"<style>\n{css}\n</style>")
    skeleton_soup.head.insert(1, f"<script>\n{js}\n</script>")
    output = skeleton_soup.prettify()
    output = output.replace("&lt;", "<").replace("&gt;", ">")
    with open("index.html", "w") as f:
        f.write(output)

start = time.time()
asyncio.run(main())
print(time.time()-start)
