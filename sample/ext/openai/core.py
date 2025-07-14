import base64
import json
import os
import asyncio
import requests
from enum import Enum
from typing import List
from agents import Agent, Runner
from openai import OpenAI
from pydantic import BaseModel


client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)


def generate_text():
    response = client.responses.create(
        model="gpt-4.1", # "gpt-4o"
        input="Write a one-sentence bedtime story about a unicorn."
    )
    print(response.output_text)

    response = client.responses.create(
        model="gpt-4o",
        instructions="Talk like a pirate.",
        input="Are semicolons optional in JavaScript?",
    )
    for output in response.output:
        print(output.model_dump_json(indent=2))

    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "developer",
                "content": "Talk like a pirate."
            },
            {
                "role": "user",
                "content": "Are semicolons optional in JavaScript?"
            }
        ]
    )
    print(response.output_text)


def analyze_image_input():
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {"role": "user", "content": "what teams are playing in this image?"},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3b/LeBron_James_Layup_%28Cleveland_vs_Brooklyn_2018%29.jpg/960px-LeBron_James_Layup_%28Cleveland_vs_Brooklyn_2018%29.jpg"
                    }
                ]
            }
        ]
    )
    print(response.output_text)


def web_search_input():
    response = client.responses.create(
        model="gpt-4.1",
        tools=[{"type": "web_search_preview"}],
        input="What was a positive news story from today?"
    )
    print(response.output_text)


def stream_event():
    stream = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": "Say 'double bubble bath' ten times fast.",
            },
        ],
        stream=True,
    )
    for event in stream:
        print(event)


async def language_agent():
    spanish_agent = Agent(
        name="Spanish agent",
        instructions="You only speak Spanish.",
    )
    english_agent = Agent(
        name="English agent",
        instructions="You only speak English",
    )
    triage_agent = Agent(
        name="Triage agent",
        instructions="Handoff to the appropriate agent based on the language of the request.",
        handoffs=[spanish_agent, english_agent],
    )
    result = await Runner.run(triage_agent, input="Hola, ¿cómo estás?")
    print(result.final_output)


def playground_prompt():
    ziea_prompt_id = "pmpt_68729abdc6948193b73cc3dfe02965490fb7bc34616d14c9"
    response = client.responses.create(
        prompt={
            "id": ziea_prompt_id,
            "version": "1",
            "variables": {
                "city": "Shanghai"
            }
        }
    )


def playground_prompt_with_file():
    ziea_prompt_id = "pmpt_68729abdc6948193b73cc3dfe02965490fb7bc34616d14c9"
    # Upload a PDF we will reference in the variables
    file = client.files.create(
        file=open("draconomicon.pdf", "rb"),
        purpose="user_data",
    )
    response = client.responses.create(
        prompt={
            "id": ziea_prompt_id,
            "version": "1",
            "variables": {
                "city": "Shanghai",
                "reference_pdf": {
                    "type": "input_file",
                    "file_id": file.id,
                },
            }
        }
    )


def makedown_prompt():
    """_summary_
    In general, a developer message will contain the following sections, usually in this order
    (though the exact optimal content and order may vary by which model you are using):
    - Identity: Describe the purpose, communication style, and high-level goals of the assistant.
    - Instructions: Provide guidance to the model on how to generate the response you want.
        What rules should it follow? What should the model do, and what should the model never do?
        This section could contain many subsections as relevant for your use case, like how the model
        should call custom functions.
    - Examples: Provide examples of possible inputs, along with the desired output from the model.
    - Context: Give the model any additional information it might need to generate a response,
        like private/proprietary data outside its training data, or any other data you know will be
        particularly relevant. This content is usually best positioned near the end of your prompt,
        as you may include different context for different generation requests.
    """
    with open("prompt-fewshot.md", "r", encoding="utf-8") as f:
        instructions = f.read()
    response = client.responses.create(
        model="gpt-4.1",
        instructions=instructions,
        input="How would I declare a variable for a last name?",
    )
    print(response.output_text)


def generate_images():
    response = client.responses.create(
        model="gpt-4.1-mini",
        input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
        tools=[{"type": "image_generation"}],
    )
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]
    if image_data:
        image_base64 = image_data[0]
        with open("cat_and_otter.png", "wb") as f:
            f.write(base64.b64decode(image_base64))


def analyze_images_onurl():
    response = client.responses.create(
        model="gpt-4.1",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what's in this image?"},
                {
                    "type": "input_image",
                    "image_url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    "detail": "high", # low, high, or auto
                },
            ],
        }],
    )
    print(response.output_text)


def analyze_images_onbase64():
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    image_path = "path_to_your_image.jpg"
    base64_image = encode_image(image_path)
    response = client.responses.create(
        model="gpt-4.1",
        input=[
            {
                "role": "user",
                "content": [
                    { "type": "input_text", "text": "what's in this image?" },
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{base64_image}",
                    },
                ],
            }
        ],
    )
    print(response.output_text)


def analyze_images_onfileid():
    def create_file(file_path):
        with open(file_path, "rb") as file_content:
            result = client.files.create(
                file=file_content,
                purpose="vision",
            )
            return result.id

    file_id = create_file("path_to_your_image.jpg")
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": "what's in this image?"},
                {
                    "type": "input_image",
                    "file_id": file_id,
                },
            ],
        }],
    )
    print(response.output_text)


def audio_output():
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": "Is a golden retriever a good family dog?"
            }
        ]
    )
    print(completion.choices[0])
    wav_bytes = base64.b64decode(completion.choices[0].message.audio.data)
    with open("dog.wav", "wb") as f:
        f.write(wav_bytes)


def audio_input():
    url = "https://cdn.openai.com/API/docs/audio/alloy.wav"
    response = requests.get(url)
    response.raise_for_status()
    wav_data = response.content
    encoded_string = base64.b64encode(wav_data).decode('utf-8')
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "wav"},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What is in this recording?"
                    },
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": encoded_string,
                            "format": "wav"
                        }
                    }
                ]
            },
        ]
    )
    print(completion.choices[0].message)


def json_schema():
    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": "Extract the event information."},
            {
                "role": "user",
                "content": "Alice and Bob are going to a science fair on Friday.",
            },
        ],
        text_format=CalendarEvent,
    )
    event = response.output_parsed
    print(event)


def chain_schema():
    class Step(BaseModel):
        explanation: str
        output: str

    class MathReasoning(BaseModel):
        steps: list[Step]
        final_answer: str

    try:
        response = client.responses.parse(
            model="gpt-4o-2024-08-06",
            input=[
                {
                    "role": "system",
                    "content": "You are a helpful math tutor. Guide the user through the solution step by step.",
                },
                {"role": "user", "content": "how can I solve 8x + 7 = -23"},
            ],
            text_format=MathReasoning,
        )
        math_reasoning = response.output_parsed
        # If the model refuses to respond, you will get a refusal message
        if (math_reasoning.refusal):
            print(math_reasoning.refusal)
        else:
            print(math_reasoning.parsed)
    except Exception as e:
        # handle errors like finish_reason, refusal, content_filter, etc.
        pass


def chain_schema2():
    response = client.responses.create(
        model="gpt-4o-2024-08-06",
        input=[
            {"role": "system", "content": "You are a helpful math tutor. Guide the user through the solution step by step."},
            {"role": "user", "content": "how can I solve 8x + 7 = -23"}
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "math_response",
                "schema": {
                    "type": "object",
                    "properties": {
                        "steps": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "explanation": {"type": "string"},
                                    "output": {"type": "string"}
                                },
                                "required": ["explanation", "output"],
                                "additionalProperties": False
                            }
                        },
                        "final_answer": {"type": "string"}
                    },
                    "required": ["steps", "final_answer"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }
    )
    print(response.output_text)


def data_extraction_schema():
    class ResearchPaperExtraction(BaseModel):
        title: str
        authors: list[str]
        abstract: str
        keywords: list[str]

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": "You are an expert at structured data extraction. You will be given unstructured text from a research paper and should convert it into the given structure.",
            },
            {"role": "user", "content": "..."},
        ],
        text_format=ResearchPaperExtraction,
    )
    research_paper = response.output_parsed
    print(research_paper)


def ui_generation_schema():
    class UIType(str, Enum):
        div = "div"
        button = "button"
        header = "header"
        section = "section"
        field = "field"
        form = "form"

    class Attribute(BaseModel):
        name: str
        value: str

    class UI(BaseModel):
        type: UIType
        label: str
        children: List["UI"]
        attributes: List[Attribute]

    UI.model_rebuild()  # This is required to enable recursive types

    class Response(BaseModel):
        ui: UI

    response = client.responses.parse(
        model="gpt-4o-2024-08-06",
        input=[
            {
                "role": "system",
                "content": "You are a UI generator AI. Convert the user input into a UI.",
            },
            {"role": "user", "content": "Make a User Profile Form"},
        ],
        text_format=Response,
    )
    ui = response.output_parsed
    print(ui)


def stream_schema():
    class EntitiesModel(BaseModel):
        attributes: List[str]
        colors: List[str]
        animals: List[str]

    with client.responses.stream(
        model="gpt-4.1",
        input=[
            {"role": "system", "content": "Extract entities from the input text"},
            {
                "role": "user",
                "content": "The quick brown fox jumps over the lazy dog with piercing blue eyes",
            },
        ],
        text_format=EntitiesModel,
    ) as stream:
        for event in stream:
            if event.type == "response.refusal.delta":
                print(event.delta, end="")
            elif event.type == "response.output_text.delta":
                print(event.delta, end="")
            elif event.type == "response.error":
                print(event.error, end="")
            elif event.type == "response.completed":
                print("Completed")
                # print(event.response.output)

        final_response = stream.get_final_response()
        print(final_response)


def json_mode():
    we_did_not_specify_stop_tokens = True
    try:
        response = client.responses.create(
            model="gpt-3.5-turbo-0125",
            input=[
                {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
                {"role": "user", "content": "Who won the world series in 2020? Please respond in the format {winner: ...}"}
            ],
            text={"format": {"type": "json_object"}}
        )

        # Check if the conversation was too long for the context window, resulting in incomplete JSON 
        if response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens":
            # your code should handle this error case
            pass

        # Check if the OpenAI safety system refused the request and generated a refusal instead
        if response.output[0].content[0].type == "refusal":
            # your code should handle this error case
            # In this case, the .content field will contain the explanation (if any) that the model generated for why it is refusing
            print(response.output[0].content[0]["refusal"])

        # Check if the model's output included restricted content, so the generation of JSON was halted and may be partial
        if response.status == "incomplete" and response.incomplete_details.reason == "content_filter":
            # your code should handle this error case
            pass

        if response.status == "completed":
            # In this case the model has either successfully finished generating the JSON object according to your schema, or the model generated one of the tokens you provided as a "stop token"

            if we_did_not_specify_stop_tokens:
                # If you didn't specify any stop tokens, then the generation is complete and the content key will contain the serialized JSON object
                # This will parse successfully and should now contain  "{"winner": "Los Angeles Dodgers"}"
                print(response.output_text)
            else:
                # Check if the response.output_text ends with one of your stop tokens and handle appropriately
                pass
    except Exception as e:
        # Your code should handle errors here, for example a network error calling the API
        print(e)


def get_weather():
    def get_weather(latitude, longitude):
        response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&current=temperature_2m,wind_speed_10m&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m")
        data = response.json()
        return data['current']['temperature_2m']

    tools = [{
        "type": "function",
        "name": "get_weather",
        "description": "Get current temperature for provided coordinates in celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "latitude": {"type": "number"},
                "longitude": {"type": "number"}
            },
            "required": ["latitude", "longitude"],
            "additionalProperties": False
        },
        "strict": True
    }]
    input_messages = [{"role": "user", "content": "What's the weather like in Paris today?"}]
    response = client.responses.create(
        model="gpt-4.1",
        input=input_messages,
        tools=tools
    )
    print(response.output)
    """output
    output = [{
        "type": "function_call",
        "id": "fc_12345xyz",
        "call_id": "call_12345xyz",
        "name": "get_weather",
        "arguments": "{\"latitude\":48.8566,\"longitude\":2.3522}"
    }]
    """
    tool_call = response.output[0]
    args = json.loads(tool_call.arguments)
    result = get_weather(args["latitude"], args["longitude"])

    input_messages.append(tool_call)  # append model's function call message
    input_messages.append({           # append result message
        "type": "function_call_output",
        "call_id": tool_call.call_id,
        "output": str(result)
    })

    response_2 = client.responses.create(
        model="gpt-4.1",
        input=input_messages,
        tools=tools,
    )
    print(response_2.output_text)


if __name__ == "__main__":
    ### Text and Prompting
    #generate_text()
    #analyze_image_input()
    #web_search_input()
    #stream_event()
    #asyncio.run(language_agent())
    #playground_prompt()
    #makedown_prompt()

    ### Images and Vision
    #generate_images()
    #analyze_images_onurl()
    #analyze_images_onbase64()
    #analyze_images_onfileid()

    ### Audio and Speech
    #audio_output()
    #audio_input()

    ### Structured Outputs
    #json_schema()
    #chain_schema()
    #chain_schema2()
    #data_extraction_schema()
    #ui_generation_schema()
    #stream_schema()
    #json_mode()

    ### Function Calling
    get_weather()
    pass

