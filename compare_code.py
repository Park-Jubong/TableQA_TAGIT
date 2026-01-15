import pandas as pd
from io import StringIO
import jsonlines
from collections import OrderedDict
import traceback


with open(f"./results/train_full_1_0_test_100_nl_sim.json", "r") as file :
    data = pd.read_json(file)

import unicodedata
data = data.applymap(lambda x: unicodedata.normalize("NFKD", x) if isinstance(x, str) else x)

answer = data["answer"]

data_1 = []
with jsonlines.open(f"./exec_results/train_full_1_test_100_code_sim.jsonl", "r") as file :
    for line in file :
        data_1.append(line["prediction"])
        
data_2 = []
with jsonlines.open(f"./exec_results/train_full_1_test_100_nl_sim.jsonl", "r") as file :
    for line in file :
        data_2.append(line["prediction"])
    
cnt1 = 0
cnt2 = 0
cnt_e = 0
for idx, (a, p1, p2) in enumerate(zip(answer, data_1, data_2)) :
    if a == p1 : 
        cnt1 = cnt1 + 1 
    if a == p2 :
        cnt2 = cnt2 + 1
    if a == p1 and a != p2 :
        print(idx+1)

print("----------")
print(cnt1)
print(cnt2)
