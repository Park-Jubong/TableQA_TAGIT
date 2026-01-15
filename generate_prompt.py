import pandas as pd
import numpy as np
from tqdm import tqdm, trange
import json
import argparse

from transformers import BertTokenizer, BertModel
import torch
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity
from sklearn.metrics import jaccard_score

import unicodedata
import csv
from io import StringIO


import tiktoken

# def read_table(context) : 
#     try : 
#         with open(f"{context}", "r") as file:
#             table = pd.read_csv(file, sep="\s*\|\s*",  engine='python')
#         table = table.drop(table.columns[[0, -1]], axis = 1)
        
#     except : 
#         with open(f"{context}", "r") as file:
#             table = pd.read_csv(file, sep="\s+\|\s", engine = 'python')
#         new_column = []
#         for i in table.columns : 
#             new_column.append(i.replace("|", "").strip())
#         table.columns = new_column
#         for i in table.columns :
#             table[i] = table[i].str.replace("|", "")
#             table[i] = table[i].str.strip().replace("", None)
            
#     return table


def read_table_2(context) : 
    try : 
        with open(f"{context}", "r") as file:
            table = pd.read_csv(file, sep="\s*\|\s*",  engine='python', encoding = "utf-8")
        table = table.drop(table.columns[[0, -1]], axis = 1)

    except : 
        with open(f"{context}", "r") as file:
            table = pd.read_csv(file, sep="\s+\|\s", engine = 'python', encoding = "utf-8")
            
    table.columns = table.columns.str.strip()
    table = table.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    table = table.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)
    
    def to_numeric(cell):
        if isinstance(cell, str):  
            no_comma = cell.replace(',', '')
            
            try:
                float(no_comma)
                return pd.to_numeric(no_comma, errors='coerce')
            except ValueError:
                return cell  
        return pd.to_numeric(cell, errors='coerce')

    table = table.applymap(to_numeric)
    
    table_txt = table.to_markdown(index = False)
    table_txt = " | ".join(i.strip() for i in table_txt.split("|")).replace("|  |", "| \n |")
    table_txt = table_txt.split("| \n |")
    del table_txt[1]
    table_txt = "| \n |".join(table_txt)
    return table, table_txt

def read_html(context) :
    try : 
        table = pd.read_html(f"{context}", encoding='utf-8')
        table = pd.DataFrame(table[0])
    except : 
        table = pd.read_html(f"{context}", encoding='utf-8')
        table = pd.DataFrame(table[0])
        columns = table.columns
        new_col = []
        for j in range(len(columns)) :
            new_col.append(columns[j][1])
        table.columns = new_col
        
    # table.columns = table.columns.str.strip()
    table = table.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    
    table = table.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)
    
    def to_numeric(cell):
        if isinstance(cell, str):  
            no_comma = cell.replace(',', '')
            
            try:
                float(no_comma)
                return pd.to_numeric(no_comma, errors='coerce')
            except ValueError:
                return cell  
        return pd.to_numeric(cell, errors='coerce')

    table = table.applymap(to_numeric)
    
    table_txt = table.to_markdown(index = False)
    table_txt = " | ".join(i.strip() for i in table_txt.split("|")).replace("|  |", "| \n |")
    table_txt = table_txt.split("| \n |")
    del table_txt[1]
    table_txt = "| \n |".join(table_txt)
    
    return table, table_txt


def self_check_prompt (data, python_data, num_shot, random_prompt) :
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    question = data["utterance"]
    tables = data["context"]
    answer = data["targetValue"]

    nl_ = python_data["question"]
    code_ = python_data["code"]
    answer_nl = python_data["answer"]
    tables_nl = python_data["table_id"]

    nl = []
    for i in nl_ :
        nl.append(i.lower())
    code = []
    for i in code_ :
        code.append(i.replace("@@@", "\n"))

    # for index, (q, t, a) in enumerate(zip(question, tables, answer)) :
    #     if q.lower() in nl :
    #         del question[index]
    #         del tables[index]
    #         del answer[index]
    # question = question.reset_index(drop=True)
    # tables = tables.reset_index(drop=True)
    # answer = answer.reset_index(drop=True)
    
    # question_check = question.copy()
    # for index, (q, t, a) in enumerate(zip(question_check, tables, answer)) :
    #     if q.lower() in nl :
    #         del question_check[index]
    #         del tables[index]
    #         del answer[index]
    # tables = tables.reset_index(drop=True)
    # answer = answer.reset_index(drop=True)
    # indexes = question_check.index
    # question_check = question_check.reset_index(drop=True)
    # print(len(indexes))
            
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')


    nl_emb = []
    for i in tqdm(nl[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        nl_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
        
    nl_emb = np.vstack(nl_emb)
        
    # q_emb = []
    # for i in tqdm(question[:]) :
    #     token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
    #     with torch.no_grad():
    #         emb = model(**token)
    #     q_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    # q_emb = np.vstack(q_emb)

    sims = cosine_similarity(nl_emb, nl_emb)

    scores = []
    for i, sim_scores in enumerate(sims):
        temp = {}
        for j_nl, sim, j_code in zip(nl, sim_scores, code) :
            temp[j_nl] = [sim, j_code]
        
        temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
        
        scores.append(temp)
        # if i in indexes :
        #     scores.append(temp)
        
    prompts = []

    
    for i in trange(0, len(nl_emb)):
    # for i in trange(0, len(question_check)) :
        data = {}
        prompt = "/* Some Python DataFrame examples are provided based on similar problems: */\n"
        few_shots = ""
        if random_prompt :
            import random
            random.seed = 123
            random.shuffle(scores[i])
        for example in scores[i][:num_shot] :
            few_shot = f"/* Answer the following: {example[0]} */\nCode : {example[1][1]}\n\n"
            few_shots = few_shots + few_shot
            
        # _, table = read_table_2(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".table")
        _, table = read_table_2(f"../WikiTableQuestions/csv/{tables_nl[i].split('/')[1]}/{tables_nl[i].split('/')[2].replace('_table', '')}")
        
        # init_table =  {"header" : header, "rows" : rows}
        # # read_table
        # table = construct_markdown_table(**init_table)
        # df = markdown_to_df(table)
        # df = convert_cells_to_numbers(df)
        # table = df.to_markdown()

        table_tok = enc.encode(str(table))
        if len(table_tok) > 3500 :
            table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
        
        test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {nl[i]} */\nCode(ONLY DataFrame Code Form) : "
        # test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {question[i]} */\nCode(ONLY DataFrame Code Form) : "
        
        prompt = prompt + few_shots + test_data
        
        data["question"] = nl[i]
        data["prompt"] = prompt
        
        data["code"] = code[i]
        data["answer"] = answer_nl[i]
        data["table"] = f"csv/{tables_nl[i].split('/')[1]}/{tables_nl[i].split('/')[2].replace('_table', '')}"

        prompts.append(data)

    with open("python_nl_sim_prompt.json", 'wt') as out:
        json.dump(prompts, out, sort_keys=True, indent=2, separators=(',', ': '))


def pyagent_prompt (data, row_selection) : 
    
    with open("./results/train_300_test_100_nl_sim.json", "r") as file :
        data = pd.read_json(file)
        
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")       
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

#     pyagent_prompt = """
# You are working with a pandas dataframe in Python. The name of the dataframe is `df`. Your task is to use `python_repl_ast` to answer the question posed to you.

# Tool description:
# - `python_repl_ast`: A Python shell. Use this to execute python commands. Input should be a valid python command. When using this tool, sometimes the output is abbreviated - ensure it does not appear abbreviated before using it in your answer.

# Guidelines:
# - **Aggregated Rows**: Be cautious of rows that aggregate data such as 'total', 'sum', or 'average'. Ensure these rows do not influence your results inappropriately.
# - **Data Verification**: Before concluding the final answer, always verify that your observations align with the original table and question.

# Strictly follow the given format to respond:

# Question: the input question you must answer
# Thought: you should always think about what to do to interact with `python_repl_ast`
# Action: can **ONLY** be `python_repl_ast`
# Action Input: the input code to the action
# Observation: the result of the action
# ... (this Thought/Action/Action Input/Observation can repeat N times)
# Thought: after verifying the table, observations, and the question, I am confident in the final answer
# Final Answer: the final answer to the original input question (AnswerName1, AnswerName2...)

# Notes for final answer:
# - Ensure the final answer format is only "Final Answer: AnswerName1, AnswerName2..." form, no other form. 
# - Ensure the final answer is a number or entity names, as short as possible, without any explanation.
# - Ensure to have a concluding thought that verifies the table, observations and the question before giving the final answer.

# You are provided with a table. This is the result of `print(df.to_markdown())` and then selected 5 rows it seemed to be helpful in solving the question. So paying particular attention to the column and detailed contents.:

# [TABLE]

# **Note**: All cells in the table should be considered as `object` data type, regardless of their appearance.

# Begin!
# Question: [QUESTION]
# """
    # q_id = data["id"]
    # db = data["context"]
    # question = data["utterance"]
    # answer = data["targetValue"]
    db = data["table"]
    question = data["question"]
    pre_prompt = data["prompt"]
    answer = data["answer"]
    
    prompts = []
    for i in trange(0, len(answer)) :
        data = {}

        t = db[i].split('.')[0] + ".table"
        _, table = read_table_2(f"../WikiTableQuestions/{t}")
        
        if row_selection :
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertModel.from_pretrained('bert-base-uncased')
            table_rows = table.split("| \n |")
            
            rows_emb = []
            for row in table_rows[1:] :                
                token = tokenizer(row, return_tensors="pt", padding=True, truncation=True)
                with torch.no_grad():
                    emb = model(**token)
                rows_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
            rows_emb = np.vstack(rows_emb)
                
            token = tokenizer(question[i], return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                emb = model(**token)
            q_emb = emb.last_hidden_state[:, 0, :].squeeze()
            q_emb = np.vstack([q_emb])

            sims = cosine_similarity(q_emb, rows_emb)
            temp = {}
            row_idx = 0
            for sim, row in zip(sims[0], table_rows[1:]) :
                temp[row_idx] = [sim, row]
                row_idx = row_idx + 1
            temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
            if len(table_rows) > 5 :
                num_row = 5
            else : 
                num_row = len(table_rows)
            table = table_rows[0]
            for row in temp[:num_row] : 
                table = table + "| \n |" + row[1][1]
            
        # table_tok = enc.encode(str(table))
        # if len(table_tok) > 3500 :
        #     table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"  
        prompt = pyagent_prompt.replace("[TABLE]", table).replace("[QUESTION]", question[i])
        
        data["question"] = question[i]
        data["prompt"] = prompt
        data["answer"] = answer[i]
        data["table"] = f"csv/{db[i].split('/')[1]}/{db[i].split('/')[2].replace('_table', '')}"

        prompts.append(data)

    with open("pyagent_prompt.json", 'wt') as out:
        json.dump(prompts, out, sort_keys=True, indent=2, separators=(',', ': ')) 
        
def keyword_sim (data, python_data, num_shot, random_prompt, n_gram) :
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    question = data["utterance"]
    tables = data["context"]
    answer = data["targetValue"]

    nl_ = python_data["question"]
    code_ = python_data["code"]
    answer_nl = python_data["answer"]
    tables_nl = python_data["table_id"]

    nl = []
    for i in nl_ :
        nl.append(i.lower())
    code = []
    for i in code_ :
        code.append(i.replace("@@@", "\n"))

    
    keywords = []
    
    from sklearn.feature_extraction.text import CountVectorizer
    from sentence_transformers import SentenceTransformer
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    
    print("Keyword")
    # key bert
    n_gram = n_gram
    for i in tqdm(question[:]) : 
        n_gram_range = (n_gram, n_gram)
        count = CountVectorizer(ngram_range = n_gram_range).fit([i])
        candidates = count.get_feature_names_out()
    
        token = tokenizer(candidates[0], return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        keyword_emb = emb.last_hidden_state[:, 0, :].squeeze()
        keywords.append((candidates[0], keyword_emb))

        
        # cnt = 0
        # for j in nl:
        #     if candidates[0] in j :
        #         print("!!!")
        #         cnt = cnt + 1
                
    #     # print(cnt) 
        
    # # question token keyword
    # for i in tqdm(question[:5]) :
    #     q_tok = i.split(" ")
        
    #     q_tok_emb = []
    #     for j in tqdm(q_tok[:]) :
    #         token = tokenizer(j, return_tensors="pt", padding=True, truncation=True)
    #         with torch.no_grad():
    #             emb = model(**token)
    #         q_tok_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    #     q_tok_emb = np.vstack(q_tok_emb)

    #     token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
    #     with torch.no_grad():
    #         emb = model(**token)
    #     q_emb = emb.last_hidden_state[:, 0, :].squeeze()
    #     q_emb = np.vstack([q_emb])

    #     sims = cosine_similarity(q_emb, q_tok_emb)
        
    #     for i, sim_scores in enumerate(sims):
    #         temp = {}
    #         for sim, tok, tok_emb in zip(sim_scores, q_tok, q_tok_emb) :
    #             temp[tok] = [sim, tok_emb]
    #     temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)

    #     keywords.append(temp[0])
            
    print("Similarity")
    nl_emb = []
    for i in tqdm(nl[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        nl_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    nl_emb = np.vstack(nl_emb)

    compare_embds = []
    for i in keywords :
        compare_embds.append(i[1])
    compare_embds = np.vstack(compare_embds)
    
    q_emb = []
    for i in tqdm(question[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        q_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    q_emb = np.vstack(q_emb)
    
    # case 1 : key word 포함, question vs nl sim
    sims = cosine_similarity(q_emb, nl_emb)
    scores = []
    total = 0
    cnt2 = 0
    for idx, sim_scores in enumerate(sims):
        temp = {}
        cnt = 0
        for j_nl, sim, j_code in zip(nl, sim_scores, code) :
            if keywords[idx][0] in j_nl : 
                temp[j_nl] = [sim, j_code, keywords[idx][0]]
                cnt = cnt + 1
            else : 
                temp[j_nl] = [0, j_code, keywords[idx][0]]
                
        if (cnt==0) :
            cnt2 = cnt2 + 1
            for j_nl, sim, j_code in zip(nl, sim_scores, code) :
                temp[j_nl] = [sim, j_code, keywords[idx][0]]
            
        temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
        
        scores.append(temp)

        
        total = total + cnt
    print(f"total: {total}")
    print(f"avg: {total/len(sims)}")
    print(f"cnt2: {cnt2}")
    
    # # case 2 : key_word vs nl sim 
    # sims = cosine_similarity(compare_embds, nl_emb)
    # scores = []
    # for idx, sim_scores in enumerate(sims):
    #     temp = {}
    #     cnt = 0
    #     for j_nl, sim, j_code in zip(nl, sim_scores, code) :
    #         temp[j_nl] = [sim, j_code, keywords[idx][0]]
            
    #     temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
    #     scores.append(temp)

        
    prompts = []
    
    for i in trange(0, len(question[:])):
        data = {}
        prompt = "/* Some Python DataFrame examples are provided based on similar problems: */\n"
        few_shots = ""
        
        if random_prompt :
            import random
            random.seed = 123
            random.shuffle(scores[i])
            
        for example in scores[i][:num_shot] :
            few_shot = f"/* Answer the following: {example[0]} */\nCode : {example[1][1]}\n\n"
            few_shots = few_shots + few_shot
            
        _, table = read_table_2(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".table")
        # _, table = read_html(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".html")
        
        table_tok = enc.encode(str(table))
        if len(table_tok) > 3500 :
            table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
        
        test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {question[i]} */\nCode(ONLY DataFrame Code Form) : "
        
        prompt = prompt + few_shots + test_data
        
        data["question"] = question[i]
        data["prompt"] = prompt
        data["answer"] = answer[i]
        data["table"] = tables[i]
        data["keyword"] = scores[i][0][1][2]

        prompts.append(data)

    with open("python_keyword_sim_prompt.json", 'wt') as out:
        json.dump(prompts, out, sort_keys=True, indent=2, separators=(',', ': '))

def nl_sim (data, python_data, num_shot, random_prompt) :
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    question = data["utterance"]
    tables = data["context"]
    answer = data["targetValue"]

    nl_ = python_data["question"]
    code_ = python_data["code"]
    answer_nl = python_data["answer"]
    tables_nl = python_data["table_id"]

    nl = []
    for i in nl_ :
        nl.append(i.lower())
    code = []
    for i in code_ :
        code.append(i.replace("@@@", "\n"))

    # for index, (q, t, a) in enumerate(zip(question, tables, answer)) :
    #     if q.lower() in nl :
    #         del question[index]
    #         del tables[index]
    #         del answer[index]
    # question = question.reset_index(drop=True)
    # tables = tables.reset_index(drop=True)
    # answer = answer.reset_index(drop=True)
            
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    nl_emb = []
    for i in tqdm(nl[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        nl_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
        
    nl_emb = np.vstack(nl_emb)
    
    q_emb = []
    for i in tqdm(question[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        q_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    q_emb = np.vstack(q_emb)
    
    sims = cosine_similarity(q_emb, nl_emb)
    # print(sims_cosine[0])
    # sims = []
    # for i in question :
    #     sim_1 = []
    #     for j in nl :
    #         temp = jaccard_score(i.split(), j.split(), average=None)
    #         sim_1.apppend(temp)
    #     sims.append(sim_1)
    # print(sims[0])
    
    
    scores = []
    for i, sim_scores in enumerate(sims):
        temp = {}
        for j_nl, sim, j_code in zip(nl, sim_scores, code) :
            temp[j_nl] = [sim, j_code]
            
        temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
        
        scores.append(temp)

    prompts = []
    
    for i in trange(0, len(q_emb)):
        data = {}
        prompt = "/* Some Python DataFrame examples are provided based on similar problems: */\n"
        few_shots = ""
        
        if random_prompt :
            import random
            random.seed = 123
            random.shuffle(scores[i])
            
        for example in scores[i][:num_shot] :
            few_shot = f"/* Answer the following: {example[0]} */\nCode : {example[1][1]}\n\n"
            few_shots = few_shots + few_shot
            
        _, table = read_table_2(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".table")
        # _, table = read_html(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".html")
        
        table_tok = enc.encode(str(table))
        if len(table_tok) > 3500 :
            table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
        
        test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {question[i]} */\nCode(ONLY DataFrame Code Form) : "
        
        prompt = prompt + few_shots + test_data
        
        data["question"] = question[i]
        data["prompt"] = prompt
        data["answer"] = answer[i]
        data["table"] = tables[i]

        prompts.append(data)

    with open("python_nl_sim_prompt.json", 'wt') as out:
        json.dump(prompts, out, sort_keys=True, indent=2, separators=(',', ': '))

        
        
def code_sim (python_data, input_data, num_shot, random_prompt) : 
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    with open(f"./results/{input_data}.json", "r") as file :
        pre_predicted = pd.read_json(file)
    
    prediction = pre_predicted["prediction"]
    question = pre_predicted["question"]
    tables = pre_predicted["table"]
    answer = pre_predicted["answer"]
    pre_prompt = pre_predicted["prompt"]
    
    nl_ = python_data["question"]
    code_ = python_data["code"]
    answer_nl = python_data["answer"]
    tables_nl = python_data["table_id"]

    nl = []
    for i in nl_ :
        nl.append(i.lower())
        
    code = []
    masked_code = []
    for i in code_ :
        code.append(i.replace("@@@", "\n"))

        masked = []
        tmp = True
        c = i.replace("@@@", "\n")
        for t in c :
            if t =="'" and tmp == True:
                tmp = False
            elif t == "'" and tmp == False : 
                masked.append("'MASK'")
                tmp = True
                
            if t != "'" and tmp == True : 
                masked.append(t)
            elif tmp == False :
                continue
            
        masked_code.append("".join(masked))
    
    masked_prediction = []
    for i in prediction : 
        masked = []
        tmp = True
        for t in i :
            if t =="'" and tmp == True:
                tmp = False
            elif t == "'" and tmp == False : 
                masked.append("'MASK'")
                tmp = True
                
            if t != "'" and tmp == True : 
                masked.append(t)
            elif tmp == False :
                continue
            
        masked_prediction.append("".join(masked))
        
    # for index, (q, t, a) in enumerate(zip(question, tables, answer)) :
    #     if q.lower() in nl :
    #         del question[index]
    #         del tables[index]
    #         del answer[index]
    # question = question.reset_index(drop=True)
    # tables = tables.reset_index(drop=True)
    # answer = answer.reset_index(drop=True)
    
            
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')


    code_emb = []
    for i in tqdm(masked_code[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        code_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
        
    code_emb = np.vstack(code_emb)
        
    pred_emb = []
    for i in tqdm(masked_prediction[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        pred_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    pred_emb = np.vstack(pred_emb)

    sims = cosine_similarity(pred_emb, code_emb)
    
    scores = []
    for i, sim_scores in enumerate(sims):
        temp = {}
        for j_nl, sim, j_code in zip(nl, sim_scores, code) :
            temp[j_nl] = [sim, j_code]
        
        temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
        
        scores.append(temp)

        
    prompts = []

    
    for i in trange(0, len(pre_predicted)):
        data = {}
        prompt = "/* Some Python DataFrame examples are provided based on similar problems: */\n"
        few_shots = ""
        
        if random_prompt :
            import random
            random.seed = 123
            random.shuffle(scores[i])
            
        for example in scores[i][:num_shot] :
            few_shot = f"/* Answer the following: {example[0]} */\nCode : {example[1][1]}\n\n"
            few_shots = few_shots + few_shot
            
        _, table = read_table_2(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".table")


        table_tok = enc.encode(str(table))
        if len(table_tok) > 3500 :
            table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
            
        q = question[i]
        test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {q} */\nCode(ONLY DataFrame Code Form) : "
        # test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {question[i]} */\nCode(ONLY DataFrame Code Form) : "
        
        prompt = prompt + few_shots + test_data
        
        data["question"] = q
        data["prompt"] = prompt
        data["pre_predicted"] = prediction[i]
        data["answer"] = answer[i]
        data["table"] = tables[i]
        
        prompts.append(data)

    with open("python_code_sim_prompt.json", 'wt') as out:
        json.dump(prompts, out, sort_keys=True, indent=2, separators=(',', ': ')) 


def dail_sim (python_data, input_data, num_shot, random_prompt) :
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")

    with open(f"./results/{input_data}.json", "r") as file :
        pre_predicted = pd.read_json(file)
    
    prediction = pre_predicted["prediction"]
    question = pre_predicted["question"]
    tables = pre_predicted["table"]
    answer = pre_predicted["answer"]
    pre_prompt = pre_predicted["prompt"]
    
    nl_ = python_data["question"]
    code_ = python_data["code"]
    answer_nl = python_data["answer"]
    tables_nl = python_data["table_id"]

    nl = []
    for i in nl_ :
        nl.append(i.lower())
        
    code = []
    masked_code = []
    for i in code_ :
        code.append(i.replace("@@@", "\n"))

        masked = []
        tmp = True
        c = i.replace("@@@", "\n")
        for t in c :
            if t =="'" and tmp == True:
                tmp = False
            elif t == "'" and tmp == False : 
                masked.append("'MASK'")
                tmp = True
                
            if t != "'" and tmp == True : 
                masked.append(t)
            elif tmp == False :
                continue
            
        masked_code.append("".join(masked))
    
    masked_prediction = []
    for i in prediction : 
        masked = []
        tmp = True
        for t in i :
            if t =="'" and tmp == True:
                tmp = False
            elif t == "'" and tmp == False : 
                masked.append("'MASK'")
                tmp = True
                
            if t != "'" and tmp == True : 
                masked.append(t)
            elif tmp == False :
                continue
            
        masked_prediction.append("".join(masked))
    
            
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # code_sim
    code_emb = []
    for i in tqdm(masked_code[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        code_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
        
    code_emb = np.vstack(code_emb)
        
    pred_emb = []
    for i in tqdm(masked_prediction[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        pred_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    pred_emb = np.vstack(pred_emb)

    sims_code = cosine_similarity(pred_emb, code_emb)
    
    # nl_sim
    nl_emb = []
    for i in tqdm(nl[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        nl_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
        
    nl_emb = np.vstack(nl_emb)
        
    q_emb = []
    for i in tqdm(question[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        q_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    q_emb = np.vstack(q_emb)

    sims_nl = cosine_similarity(q_emb, nl_emb)
    
    # code_sim threshold and sort by nl_sim
    scores = []
    threshold = 0.9

    for sim_nl, sim_code in zip(sims_nl, sims_code):
        temp = {}
        for j_nl, sim_c, sim_n, j_code in zip(nl, sim_code, sim_nl, code) :
            if sim_c > threshold :
                temp[j_nl] = [sim_n, j_code]
            else : 
                temp[j_nl] = [0, j_code]
                
        temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
        
        scores.append(temp)

        
    prompts = []

    
    for i in trange(0, len(pre_predicted)):
        data = {}
        prompt = "/* Some Python DataFrame examples are provided based on similar problems: */\n"
        few_shots = ""
        
        if random_prompt :
            import random
            random.seed = 123
            random.shuffle(scores[i])
            
        for example in scores[i][:num_shot] :
            few_shot = f"/* Answer the following: {example[0]} */\nCode : {example[1][1]}\n\n"
            few_shots = few_shots + few_shot
            
        _, table = read_table_2(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".table")


        table_tok = enc.encode(str(table))
        if len(table_tok) > 3500 :
            table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
            
        q = question[i]
        test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {q} */\nCode(ONLY DataFrame Code Form) : "
        # test_data = f"/* Given the following table: */\n{table}\n\n/* Answer the following: {question[i]} */\nCode(ONLY DataFrame Code Form) : "
        
        prompt = prompt + few_shots + test_data
        
        data["question"] = q
        data["prompt"] = prompt
        data["pre_predicted"] = prediction[i]
        data["answer"] = answer[i]
        data["table"] = tables[i]
        
        prompts.append(data)

    with open("dail_sim_prompt.json", 'wt') as out:
        json.dump(prompts, out, sort_keys=True, indent=2, separators=(',', ': '))


def table_sim(data, python_data, num_shot, random_prompt) : 
    enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    
    question = data["utterance"]
    tables = data["context"]
    answer = data["targetValue"]

    nl_ = python_data["question"]
    code_ = python_data["code"]
    answer_nl = python_data["answer"]
    tables_nl = python_data["table_id"]
    
    tables_nl_set = set(tables_nl)
    train_tab_sentences = []
    for i in tables_nl_set :
        _, table = read_table_2(f"../WikiTableQuestions/{i.split('.')[0]}"+".table")
        train_tab_sentences.append("| \n |".join(table.split("| \n |")[:2]))
    
    test_tab_sentences = []
    for i in tables :
        _, table = read_table_2(f"../WikiTableQuestions/{i.split('.')[0]}"+".table")
        test_tab_sentences.append("| \n |".join(table.split("| \n |")[:2]))
    
    nl = []
    for i in nl_ :
        nl.append(i.lower())
    code = []
    for i in code_ :
        code.append(i.replace("@@@", "\n"))
            
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    train_tab_emb = []
    for i in tqdm(train_tab_sentences[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        train_tab_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    train_tab_emb = np.vstack(train_tab_emb)
    
    test_tab_emb = []
    for i in tqdm(test_tab_sentences[:]) :
        token = tokenizer(i, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            emb = model(**token)
        test_tab_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
    test_tab_emb = np.vstack(test_tab_emb)
    
    sims = cosine_similarity(test_tab_emb, train_tab_emb)
    
    scores = []
    for i, sim_scores in enumerate(sims):
        temp = {}
        for tab, sim in zip(tables_nl, sim_scores) :
            temp[tab] = [sim]
            
        temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
        
        scores.append(temp)

    prompts = []
    
    for i in trange(0, len(question)):
        data = {}
        prompt = """
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

Here's part of an example table, question and answer :

Table : [EX TABLE]
Question : [EX Question]
Answer : [EX Answer]

And then you are provided with a table. This is the result of `print(df.to_markdown())` :

[TABLE]

**Note**: All cells in the table should be considered as `object` data type, regardless of their appearance.

Begin!
Question: [QUESTION]"""
        
        if random_prompt :
            import random
            random.seed = 123
            random.shuffle(scores[i])
                    
        _, ex_table = read_table_2(f"../WikiTableQuestions/{scores[i][0][0].split('.')[0]}"+".table")
        ex_table = "| \n |".join(ex_table.split("| \n |")[:6])
        
        check_table = [idx for idx, table in enumerate(tables_nl) if table == scores[i][0][0]]
        compare_q = []
        compare_a = []
        for idx in check_table:
            compare_q.append(nl[idx])
            compare_a.append(list(answer_nl)[idx])
        
        q_emb = []
        for j in compare_q[:] :
            token = tokenizer(j, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                emb = model(**token)
            q_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
        q_emb = np.vstack(q_emb)
        
        nl_emb = []
        test_q = [question[i]]
        for j in test_q[:] :
            token = tokenizer(j, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                emb = model(**token)
            nl_emb.append(emb.last_hidden_state[:, 0, :].squeeze())
        nl_emb = np.vstack(nl_emb)
        
        q_sims = cosine_similarity(nl_emb, q_emb)
        
        
        temp_scores = []
        for j, sim_scores in enumerate(q_sims):
            temp = {}
            for sim, _q, _a in zip(sim_scores, compare_q, compare_a) :
                temp[_q] = [sim, _q, _a]
                
            temp = sorted(temp.items(), key=lambda item: item[1][0], reverse=True)
            
            temp_scores.append(temp)

        _, table = read_table_2(f"../WikiTableQuestions/{tables[i].split('.')[0]}"+".table")

        table_tok = enc.encode(str(table))
        if len(table_tok) > 3500 :
            table = enc.decode(table_tok[:3500]) + "\n*syncopation*\n"
            
        prompt = prompt.replace("[EX TABLE]", ex_table).replace("[EX Question]", temp_scores[0][0][0]).replace("[EX Answer]", temp_scores[0][0][1][2]).replace("[TABLE]", table).replace("[QUESTION]", question[i])
        
        
        
        data["question"] = question[i]
        data["prompt"] = prompt
        data["answer"] = answer[i]
        data["table"] = tables[i]

        prompts.append(data)

    with open("python_table_sim_prompt.json", 'wt') as out:
        json.dump(prompts, out, sort_keys=True, indent=2, separators=(',', ': '))

            
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_type', type=str, default = "test",
                        help="choose prompt type")
    parser.add_argument('--input_data', type=str, default = "test",
                        help="input data for test")
    parser.add_argument('--few_shot_num', type=int, default = 1,
                        help="number of few shot example")
    parser.add_argument('--n_gram', type=int, default = 1,
                        help="number of keyword token")
    args = parser.parse_args()
    

    with open("../WikiTableQuestions/data/pristine-unseen-tables.tsv", "r") as file :
        data = pd.read_csv(file, sep = "\t")
    data = data.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

    with open("./train_data_full_table.json", "r") as file :
        python_data = pd.read_json(file)
    python_data = python_data.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)
    train_data = python_data[python_data["correct"].isin([1])]
        
    num_shot = args.few_shot_num
    print(args.prompt_type)
    if args.prompt_type == "nl_sim" : 
        nl_sim(data, train_data, num_shot, random_prompt=False)
    elif args.prompt_type == "pyagent" :
        pyagent_prompt(data, row_selection = False)
    elif args.prompt_type == "keyword_sim" :
        keyword_sim(data, train_data, num_shot, random_prompt=False, n_gram = args.n_gram)
    elif args.prompt_type == "code_sim" :
        code_sim(train_data, args.input_data, num_shot, random_prompt=False)
    elif args.prompt_type == "dail_sim" :
        dail_sim(train_data, args.input_data, num_shot, random_prompt=False)
    elif args.prompt_type == "self_check" :
        self_check_prompt(data, train_data, num_shot, random_prompt=False)
    elif args.prompt_type == "table_sim" : 
        table_sim(data, train_data, num_shot, random_prompt =False)
        
if __name__ == '__main__':
    main() 
        
 