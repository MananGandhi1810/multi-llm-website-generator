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
import aiohttp
import base64

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
worker_clients: list[Client] = [genai.Client(api_key=os.getenv("GEMINI_API_KEY_"+str(i))) for i in range(3)]
cf_ai_model = "@cf/black-forest-labs/flux-1-schnell"
cf_ai_url = f"https://api.cloudflare.com/client/v4/accounts/{os.getenv("CF_ACCOUNT_ID")}/ai/run/{cf_ai_model}"

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

class ImagePromptResponse(BaseModel):
    """
    Response model for generating images for the website.
    prompt: Prompt for the image to be generated. Should be very detailed.
    filename: Name of the file where this image should be saved and can be referenced in the website.
    """
    prompt: str
    filename: str

class WorkerResponse(BaseModel):
    """
    Response Model for the workers creating sections of the website.
    html_code: The HTML code for the section the worker is generating
    css_code: Any optional CSS. Do not include any tags, only CSS
    js_code: Any optional Javascript. Do not include any tags, only Javascript
    """
    image_prompts: list[ImagePromptResponse]
    html_code: str
    css_code: Optional[str]
    js_code: Optional[str]


async def generate_image(prompt, filename, session):
    response = await session.post(cf_ai_url, headers={"Authorization": f"Bearer {os.getenv("CF_API_KEY")}"}, json={"prompt": prompt})
    response = await response.json()
    if not response['success']:
        print(f"Could not generate image for: {prompt}")
        return
    with open(f"static/{filename}", "wb") as f:
        f.write(base64.decodebytes(response['result']["image"].encode()))

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
        Add images to the image_prompts, and files should only be png file.
        All images will be saved to /static/{"{filename}"}
        Make the design look modern and futuristic.
        Include Custom JS and CSS for that section if needed.
        Add interactivity in the elements if needed.
        Also include any custom font.
        Add micro transitions in the hero sections.
        Avoid adding multiple images to a section if not needed.
        """,
        config={
            "response_mime_type": "application/json",
            "response_schema": WorkerResponse,
        }
    )
    return (section_name, worker_response.parsed.html_code, worker_response.parsed.css_code, worker_response.parsed.js_code, worker_response.parsed.image_prompts)

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
        Include sizings for each section in the skeleton code, and share them in the prompt as well.
        DO NOT ADD ANY CODE EXCEPT BOILERPLATE/SKELETON CODE.
        Make sure you set the margins and padding to the body correctly.
        Use Tailwind for styling.
        This is the tag for TailwindCSS: <script src="https://unpkg.com/@tailwindcss/browser@4"></script>
        Include the Tailwind import tag in the skeleton.
        Put all repetitive information into the shared context.
        Explain the website in detail in the shared context.
        Set a font if needed.
        Add website style, colours, font theming, font colours, etc in the theme context.
        The prompt should be very DETAILED, and all the sections of the website should be very CONSISTENT.
        Also ask workers to add micro interactions and transitions to the hero elements.
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
        print(f"Prompt ({plan_item.section_name}): {plan_item.prompt}")
        tasks.append(generate_section(plan_item.prompt, plan_response.parsed.shared_context, plan_response.parsed.theme_context, plan_item.section_name, i))
    results = await asyncio.gather(*tasks)
    collected_image_prompts = []
    css = []
    js = []
    aiohttp_session = aiohttp.ClientSession()
    for section, html_snippet, css_snippet, js_snippet, image_prompts in results:
        collected_image_prompts.extend([generate_image(i.prompt, i.filename, aiohttp_session) for i in image_prompts])
        css.append(css_snippet)
        js.append(js_snippet)
        skeleton_soup.find(id=section).replace_with(html_snippet)
    image_generation_requests = asyncio.gather(*collected_image_prompts)
    css = "\n".join([x for x in css if x is not None])
    js = "\n".join([x for x in js if x is not None])
    skeleton_soup.head.insert(1, f"<style>\n{css}\n</style>")
    skeleton_soup.head.insert(1, f"<script>\n{js}\n</script>")
    output = skeleton_soup.prettify()
    output = output.replace("&lt;", "<").replace("&gt;", ">")
    with open("index.html", "w") as f:
        f.write(output)
    await image_generation_requests
    await aiohttp_session.close()

start = time.time()
asyncio.run(main())
print(time.time()-start)
