import pandas as pd
from io import StringIO
import jsonlines
from collections import OrderedDict
import traceback
from tqdm import tqdm

import argparse
import unicodedata

import os.path
from os import path
    

def check_answer(prediction_file, answers_file) : 

    if path.exists(f"exec_results/{answers_file}.jsonl") :
        os.remove(f"exec_results/{answers_file}.jsonl")

    with open(f"./results/{prediction_file}.json", "r") as file :
        python_data = pd.read_json(file)
    python_data = python_data.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

    answer = python_data["answer"]
    predictions = python_data["prediction"]
    table = python_data["table"]

    # io_buffer = StringIO()

    for a, p, t in zip(tqdm(answer), predictions, table) :
        
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
            
        except Exception as e :
            pred = OrderedDict()
            pred["prediction"] = str(traceback.format_exc())
            with jsonlines.open(f'exec_results/{answers_file}.jsonl', 'a') as file:
                file.write(pred)
        
        
            
    file.close()
    pred_results = []

    with jsonlines.open(f'exec_results/{answers_file}.jsonl') as file:
        for line in file.iter() :
            pred_results.append(line["prediction"])
            
    cnt = 0
    for p, a in zip(pred_results, answer) :
        # if p == "False" or p == 'false' :
        #     p = "no"
        # if p == "True" or p == 'true' :
        #     p = "yes"
        # if len(p.split(", ")) > 2 :
        #     p = p.replace(", ", "|")
        # if "dtype: object" in p :
        #     p = " ".join(p.split("\n")[0].split(" ")[1:])
        # if '\"' in p :
        #     p = p.replace("\"", "")
        # if "[" in p and "]" in p:
        #     p = p.replace("[", "").replace("]", "").replace("'", "")
            
        if str(a).strip() == str(p).strip() :
            cnt = cnt + 1
    print(cnt)


# from generate_prompt import read_table
# from collections import OrderedDict
# import pandas as pd
# import jsonlines
# df = read_table('../WikiTableQuestions/csv/204-csv/475.table')
# answer = df['Wrestler:'].value_counts().idxmax()
# print(answer)

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--prediction_file', type=str, default = "train_data_300_test_10_code_sim_sample",
                        help="choose prompt type")
    parser.add_argument('--answers_file', type=str, default = "train_data_300_test_10_code_sim_sample",
                        help="choose prompt type")

    args = parser.parse_args()
    
    # prediction_file = "train_data_300_test_10_code_sim_sample"
    # answers_file = "train_data_300_test_10_code_sim_answer"
    check_answer(args.prediction_file, args.answers_file)
        
        
if __name__ == '__main__':
    main() 
