import pandas as pd
import openai

import json
from tqdm import tqdm
from time import time

import random
import argparse

secret_key = 'Your api key'
openai.api_key = secret_key
openai.organization = 'your organization id'

def iterative_prompting(prompt, prompt_type, t, p) :
    import os.path
    from os import path
    from collections import OrderedDict
    import traceback
    import jsonlines

    answers_file = "iterative_prompting_temp"

    if path.exists(f"exec_results/{answers_file}.jsonl") :
        os.remove(f"exec_results/{answers_file}.jsonl")
        
    cnt_iter = 0
    while True : 
        try :
            t = t.split('.')[0] + ".table"

            code = f'''
from generate_prompt import read_table_2
from collections import OrderedDict
import pandas as pd
import numpy as np
import re
import jsonlines
df, _ = read_table_2('../WikiTableQuestions/{t}')
pred = OrderedDict()
'''
            prediction = str(p).strip().split("\n")
            for i, _ in enumerate(prediction) :
                if "Output" in _ :
                    del prediction[i]
        
            if len(prediction) > 1 :
                for i in range(0, len(prediction)-1) : 
                    code = code + f"{prediction[i]}\n"
                code = code + f'''pred["prediction"] = str({prediction[-1]})'''
                code = code + f'''
with jsonlines.open('exec_results/{answers_file}.jsonl', 'a') as file:
    file.write(pred)'''
            else :
                if " = " in p :
                    var_name = p.split(" = ")[0]
                    action_code = str(p).strip()
                    code = code + f'''
{action_code}
pred["prediction"] = str({var_name})
with jsonlines.open('exec_results/{answers_file}.jsonl', 'a') as file:
    file.write(pred)'''
                else : 
                    code = code + f'''
pred["prediction"] = str({p}).strip()
with jsonlines.open('exec_results/{answers_file}.jsonl', 'a') as file:
    file.write(pred)'''
    
            exec(code)
            
            return prompt, p
            
        except Exception as e :
            pred = OrderedDict()
            pred["prediction"] = str(traceback.format_exc())
            with jsonlines.open(f'exec_results/{answers_file}.jsonl', 'a') as file:
                file.write(pred)
            prompt = prompt + p + "\nExecution result (Error message) : " + str(pred["prediction"]) + "\n\nModify the code by referring to the error message.\nFixed Code(ONLY DataFrame Code Form) : "
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
                    p = choices['message']['content']
                    
                    if prompt_type == "pyagent" or prompt_type == "table_sim": 
                        p = p.split("Observation:")[0].strip()
                    break
                
                except :
                    time.sleep(3)
                    
            cnt_iter = cnt_iter + 1
            
            if cnt_iter == 3 :
                return prompt, p

def ask_llm (prompt_type, num_samples, output_file_name, iterative) :
    if prompt_type == "nl_sim" : 
        file_name = "python_nl_sim_prompt.json"
    elif prompt_type == "code_sim" :
        file_name = "python_code_sim_prompt.json"
    elif prompt_type == "keyword_sim" :
        file_name = "python_keyword_sim_prompt.json"
    elif prompt_type == "pyagent" :
        file_name = "pyagent_prompt.json"
    elif prompt_type == "dail_sim" :
        file_name = "dail_sim_prompt.json"
    elif prompt_type == "table_sim" :
        file_name = "python_table_sim_prompt.json"
    with open(file_name, "r") as file :
        python_data = pd.read_json(file)

    
    # if prompt_type == "nl_sim" or "pyagent":
    if prompt_type == "nl_sim" or prompt_type == "keyword_sim" or prompt_type =="table_sim":
        python_data = python_data.sample(frac=1, random_state = 123).reset_index(drop=True)[:num_samples]
    print(python_data)

    prompts = python_data["prompt"]
    questions = python_data["question"]
    answer = python_data["answer"]
    table = python_data["table"]


    replies = []
    new_prompts = []
    for prompt, t in zip(tqdm(prompts), table) :
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
                
                
                if prompt_type == "pyagent" or prompt_type == "table_sim": 
                    if "Action Input:" in reply : 
                        reply = reply.split("Action Input:")[1].split("Observation:")[0].strip()
                break
            
                
            except :
                time.sleep(3)
                
        if iterative : 
            prompt, reply = iterative_prompting(prompt, prompt_type, t, reply)
    
        replies.append(reply)
        new_prompts.append(prompt)
    content = []

    # code = python_data["code"]
    # for a, p, t, r, c in zip(answer, prompts, table, replies, code) :
    #     content.append({"answer" : a, "prompt" : p, "table" : t, "prediction" : r, "code" : c})
        
    for a, q, p, t, r in zip(answer, questions, new_prompts, table, replies) :
        content.append({"answer" : a, "question" : q, "prompt" : p, "table" : t, "prediction" : r})
        
    with open(f"./results/{output_file_name}.json", 'wt') as out:
        json.dump(content, out, sort_keys=True, indent=2, separators=(',', ': '))



# def pandas_ai() :
        
        
       
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', type=str, default = "test",
                        help="choose prompt type")
    parser.add_argument('--num_samples', type=int, default = 100,
                        help="choose prompt type")
    parser.add_argument('--output_file_name', type=str, default = "train_data_300_test_10_code_sim_sample",
                        help="choose prompt type")
    
    args = parser.parse_args()
    
    
    ask_llm(args.prompt_type, args.num_samples, args.output_file_name, iterative = False)
        
        
if __name__ == '__main__':
    main() 