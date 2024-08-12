from typing import List, Optional

from dotenv import load_dotenv
from openai import OpenAI
import os
import json

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import ValidationError

load_dotenv()
model_name = os.getenv("MODEL_NAME")
unify_url = os.getenv("UNIFY_URL")
unify_api_key = os.getenv("UNIFY_API_KEY")

client = OpenAI(base_url=unify_url, api_key=unify_api_key)
from pydantic import BaseModel


class AutoTag(BaseModel):
    moods: str
    genre: str
    gender: str
    other: str


def auto_tag():
    return {
        "type": "function",
        "function": {
            "name": "provide_auto_tag",
            "description": "Auto tag this song for me.",
            "parameters": {
                "type": "object",
                "properties": {
                    "moods": {
                        "type": "string",
                        "description": "moods in the song, comma separated",
                    },
                    "genre": {
                        "type": "string",
                        "description": "genre(s)",
                    },
                    "gender": {
                        "type": "string",
                        "description": "gender of the artist",
                    },
                    "other": {
                        "type": "string",
                        "description": "other useful or relevant tags",
                    },
                },
                "required": ["moods", "genre", "gender", "other"],
            },
        },
    }


def provide_auto_tag(data: AutoTag):
    return data


def cv_cmd(song):
    prompt = f"Auto tag this song for me: {song}."
    return prompt


tools_list = [auto_tag()]

fschemas = {
    "provide_auto_tag": AutoTag,
}

tools_schema = {
    "provide_auto_tag": provide_auto_tag,
}


def get_tool_call(command: str) -> Optional[List[ChatCompletionMessageToolCall]]:
    messages = [{"role": "user", "content": command}]
    tools = tools_list
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=tools,
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls
    return tool_calls


def call_tool(tool_calls: Optional[List[ChatCompletionMessageToolCall]]):
    if tool_calls:
        response = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_to_call = tools_schema[function_name]
            function_args = json.loads(tool_call.function.arguments)
            function_schema = fschemas[function_name]
            try:
                function_data_param = function_schema.parse_obj(function_args)
            except ValidationError as e:
                raise e
            function_response = function_to_call(
                function_data_param
            )
            response.append(function_response)
        return response
    else:
        raise "No tool_calls found"


def main(song: str):
    command = cv_cmd(song)
    tools = get_tool_call(command)
    response = call_tool(tools)
    return response


if __name__ == '__main__':
    import csv
    import os
    import pandas as pd
    from tqdm import tqdm

    # Replace 'your_file.xlsx' with the path to your Excel file
    df = pd.read_csv('tags/audiolm_dataset2.csv')

    # Initialize the dictionary
    songs_dict = {}
    headers = ['audio', 'caption']
    with open('audiolm_dataset_716.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        # Iterate over rows in the DataFrame
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc='Processing Rows'):
            song_path = row['audio']
            caption = row['caption']
            try:
                # Split the song path by backslashes to get categories
                parts = song_path.split('/')
                song = parts[1]
                output: AutoTag = main(song)[0]
                delimiter = ', '
                tag = delimiter.join([
                    output.moods,
                    output.genre,
                    output.gender,
                    output.other
                ])
                result = [song_path, f"{tag}"]
                writer.writerow(result)
            except:
                headers = ['audio']
                with open('audiolm_logs.csv', 'w', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(headers)
                    writer.writerow(song_path)
                continue
