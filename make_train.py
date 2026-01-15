import jsonlines
import json
import pandas as pd
import openai
import time
from generate_prompt import read_table_2
from collections import OrderedDict
import traceback
from io import StringIO
from tqdm import tqdm, trange
import tiktoken


pyagent_prompt = """
You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question posed to you.

Tool description:
- `python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes the output is abbreviated - ensure it does not appear abbreviated before using it in your answer.

Guidelines:
- **Aggregated Rows**: Be cautious of rows that aggregate data such as 'total', 'sum', or 'average'. Ensure these rows do not influence your results inappropriately.
- **Data Verification**: Before concluding the final answer, always verify that your observations align with the original table and question.

Strictly follow the given format to respond:

Question: the input question you must answer
Thought: you should always think about what to do to interact with `python_repl_ast`
Action: can **ONLY** be `python_repl_ast`
Action Input: the input code to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: after verifying the table, observations, and the question, I am confident in the final answer
Final Answer: the final answer to the original input question (AnswerName1, AnswerName2...)

Notes for final answer:
- Ensure the final answer format is only "Final Answer: AnswerName1, AnswerName2..." form, no other form. 
- Ensure the final answer is a number or entity names, as short as possible, without any explanation.
- Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

You are provided with a table. This is the result of `print(df.to_markdown())`:

[TABLE]

**Note**: All cells in the table should be considered as `object` data type, regardless of their appearance.

Begin!
Question: [QUESTION]
"""

def ask_llm(prompt) :
    secret_key = 'your api key'
    openai.api_key = secret_key
    openai.organization = 'your organization id'
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
            
        except Exception as e:
            print(e)
            time.sleep(3)
    return reply
           

with open("../WikiTableQuestions/data/training.tsv", "r") as file :
    data = pd.read_csv(file, sep = "\t")
    
import unicodedata
data = data.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

data = data.sample(frac=1, random_state=123)

q_id = data["id"]
db = data["context"]
question = data["utterance"]
answer = data["targetValue"]

cnt = 0
content = []

results = []
answers_file = "train_gold"

# for i in trange(0, len(question)) :
for i in trange(0, 1000) :
    t = db[i].split('.')[0] + ".table"
    table_df, table = read_table_2(f"../WikiTableQuestions/{t}")
    
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    table_tok = enc.encode(str(table))
    if len(table_tok) > 3500 :
        table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
        
    prompt = pyagent_prompt.replace("[TABLE]", table).replace("[QUESTION]", question[i])
    
    # ask_llm
    reply = ask_llm(prompt)
            
    # execute
    cnt_try = 1
    while True :
        try :
            answers_file = "train_gold"
            code = f'''
from generate_prompt import read_table_2
from collections import OrderedDict
import pandas as pd
import numpy as np
import jsonlines
df, _ = read_table_2('../WikiTableQuestions/{t}')
pred = OrderedDict()
'''
            action_input = reply.split("Action Input: ")[1].split("Observation: ")[0].strip()
            if "```python" in action_input : 
                action_input = action_input.replace("```python", "").replace("```", "")
            prediction = str(action_input).strip().split("\n")
            for idx, _ in enumerate(prediction) :
                if "Output" in _ :
                    del prediction[idx]
            if len(prediction) > 1 :
                for idx in range(0, len(prediction)-1) :
                    code = code + f"{prediction[idx]}\n"
                code = code + f'''pred["prediction"] = str({prediction[-1]})'''
                code = code + f'''
with jsonlines.open('./{answers_file}.jsonl', 'a') as file:
    file.write(pred)'''
            else :
                if " = " in action_input :
                    var_name = action_input.split(" = ")[0]
                    action_code = str(action_input).strip()
                    code = code + f'''
{action_code}
pred["prediction"] = str({var_name})
with jsonlines.open('./{answers_file}.jsonl', 'a') as file:
    file.write(pred)'''
                else : 
                    code = code + f'''
pred["prediction"] = str({action_input}).strip()
with jsonlines.open('./{answers_file}.jsonl', 'a') as file:
    file.write(pred)'''
    
            exec(code)
            
        except Exception as e :
            pred = OrderedDict()
            pred["prediction"] = str(traceback.format_exc())
            with jsonlines.open(f'./{answers_file}.jsonl', 'a') as file:
                file.write(pred)
        
        if str(answer[i]) == str(pred["prediction"]) or cnt_try == 1:
            break
        else :
            cnt_try = cnt_try + 1
            time.sleep(1)
            reply = ask_llm(prompt)
        


    db_id = []
    questions = []
    answers = []
    query = []
    sql = []
    predictions = []
    total_reply = []
    correct = []
    prompts = []
    # if str(answer[i]) == str(pred["prediction"]) :
    #     cnt = cnt + 1
    
    prediction = pred["prediction"]
    pred_code = action_input

    questions.append(question[i])
    answers.append(answer[i])
    db_id.append(t)
    query.append(pred_code)
    predictions.append(prediction)
    sql.append({})
    total_reply.append(reply)
    prompts.append(prompt)
    if str(answer[i]) == str(pred["prediction"]) :
        correct.append(1)
        results.append(1)
    elif str(answer[i]) != str(pred["prediction"]) and "Traceback" not in str(pred["prediction"]) :
        correct.append(0)
        results.append(0)
    else : 
        correct.append(-1)
        results.append(-1)

    if len(results) % 50 == 0 :
        print(f"#{len(results)} 1 : {results.count(1)}, 0 : {results.count(0)}, -1 : {results.count(-1)}")
    
    for d, que, quest, pro, a, p, c, re in zip(db_id, query, questions, prompts, answers, predictions, correct, total_reply) :
        content.append({"table_id" : d, "code" : que, "question" : quest, "prompt":pro, "answer" : a, "prediction" : p, "correct" : c, "reply" : re})

    with open("./train_data_1000_html.json", 'wt') as out:
        json.dump(content, out, indent=2, separators=(',', ': '))
        
    # if cnt == 300:
    #     break