import pandas as pd
from io import StringIO
import jsonlines
from collections import OrderedDict
import traceback
from tqdm import tqdm
from generate_prompt import read_table_2

import openai
import json
from time import time

secret_key = 'your api key'
openai.api_key = secret_key
openai.organization = 'your organization id'


with open(f"./results/no_iter_prompting_nl_code_test.json", "r") as file :
    data = pd.read_json(file)

import unicodedata
data = data.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)


answers = data["answer"]
predictions = data["prediction"]
prompts = data["prompt"]
questions = data["question"]
tables = data["table"]
results = []
with jsonlines.open(f"./exec_results/no_iter_prompting_nl_code_test.jsonl", "r") as file :
    for line in file :
        results.append(line["prediction"])
        
        
        
post_prompt = '''There's the table and a related shot-answer question.

[TABLE]

Question : [QUESTION]

***
To answer this question, a python dataframe code wrote. The code and execution result are as follows.
Python dataframe code : [CODE]
Execution Result : [RESULT]
***

**Please refer to this and answer the question. (NOT a sentence, Just a Simple and Short Answer)**
Answer : '''
import tiktoken
enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

post_prompts = []
for pred, p, q, t, r in zip(predictions, prompts, questions, tables, results) :
    
    _, table = read_table_2(f"../WikiTableQuestions/{t.split('.')[0]}"+".table")
        
    table_tok = enc.encode(str(table))
    if len(table_tok) > 3500 :
        table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
    prompt = post_prompt.replace("[TABLE]", table).replace("[QUESTION]", q).replace("[CODE]", pred).replace("[RESULT]", r)
    post_prompts.append(prompt)
    
    


replies = []
output_file_name = "post_process_test"
for prompt in tqdm(post_prompts) :
    while True :
        try :
            # Try Start 
            # ChatGPT
            query_chat = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo",
                messages = [
                    {"role" : "user", "content" : prompt}
                ]
            )
            
            choices = query_chat['choices'][0]
            reply = choices['message']['content']
            

            break
        
            
        except :
            time.sleep(3)
            

    replies.append(reply)
    content = []

    
    # if reply == "False" or reply == 'false' :
    #     reply = "no"
    # if reply == "True" or reply == 'true' :
    #     reply = "yes"
    # if len(reply.split(",")) > 2 :
    #     reply = reply.replace(", ", "|")


    # code = python_data["code"]
    # for a, p, t, r, c in zip(answer, prompts, table, replies, code) :
    #     content.append({"answer" : a, "prompt" : p, "table" : t, "prediction" : r, "code" : c})
        
    for a, q, p, t, r in zip(answers, questions, post_prompts, tables, replies) :
        content.append({"answer" : a, "question" : q, "prompt" : p, "table" : t, "prediction" : r})
        
    with open(f"./results/{output_file_name}.json", 'wt') as out:
        json.dump(content, out, sort_keys=True, indent=2, separators=(',', ': '))

        
