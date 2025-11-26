from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig
import shutil

pr = '''
Summary Task: \n
Describe the people in the picture and extract characteristics descriving the people according to the following descriptive format: \n

Example1: \n

Attire: Wearing a blue police uniform \n
Location Identifier: Positioned beside a white truck \n
Geographical Reference Point: Awaiting at the traffic light \n
Guidelines for Similar Text Descriptions:A passenger wearing a black short-sleeved shirt and carrying a backpack, standing on the roadside. \n

descriptive format: A passenger [Attire] [Location Identifier] [Geographical Reference Point]. \n

To effectively summarize similar passages, focus on extracting these essential details: \n

Distinctive Clothing or Accessories: Including uniforms, distinctive headwear, eyewear, or other personal items. \n
Physical Descriptive Features: Height, hair color, eye color, or any notable facial features. \n
Items Being Carried: Backpacks, briefcases, luggage, shopping bags, or any objects that might be relevant. \n
Current Actions: Gestures, activities like speaking on the phone, reading, or interacting with others. \n
Surrounding Environment Description: Buildings, vehicles, landscape, weather conditions, or urban settings. \n
Position Relative to Landmarks or Fixed Objects: Near trees, by public amenities, or close to architectural landmarks. \n'
'''

def vlm(model,tokenizer,link):

    query = tokenizer.from_list_format([
        {'image': link},
        {'text': pr},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)

    print("response: ",response)
    return response
    

