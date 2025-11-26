from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.generation import GenerationConfig

pr = '''
Summary Task: \n
Analyze the sentence enclosed within the 'text' tag and extract characteristics descriving the passenger according to the following descriptive format: \n

Example: \n

Occupation/Identity: Traffic Policeman \n
Attire: Wearing a police uniform \n
Location Identifier: Positioned beside a white truck \n
Geographical Reference Point: Awaiting at the traffic light \n
characteristics descriving: A passenger next to a white truck, visible at the traffic light, and a traffic policeman in a police uniform. \n 

descriptive format: A [Occupation/Identity] [Attire] [Location Identifier] [Geographical Reference Point]. \n

To effectively summarize similar passages, focus on extracting these essential details: \n

Occupational or Identity Indicators: Such as professional titles (doctor, teacher) or roles (student, tourist). \n
Distinctive Clothing or Accessories: Including uniforms, distinctive headwear, eyewear, or other personal items. \n
Physical Descriptive Features: Height, hair color, eye color, or any notable facial features. \n
Items Being Carried: Backpacks, briefcases, luggage, shopping bags, or any objects that might be relevant. \n
Current Actions: Gestures, activities like speaking on the phone, reading, or interacting with others. \n
Surrounding Environment Description: Buildings, vehicles, landscape, weather conditions, or urban settings. \n
Position Relative to Landmarks or Fixed Objects: Near trees, by public amenities, or close to architectural landmarks. \n
By meticulously incorporating these elements into your summary, you'll craft a vivid and accurate portrayal that aids in readily identifying or locating the subject. \n

Text: \n
'''

def summary(model,tokenizer,text):

    query = pr + text

    response, history = model.chat(tokenizer, query=query, history=None)
    
    return response
    

