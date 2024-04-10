import os
import re
from random import shuffle
import pandas as pd
import requests

queies_df = pd.read_csv(os.path.join("data", "exp_supp_queries.csv"), sep="\t")
print(queies_df)
print(queies_df.info())

# queies_dicts = queies_df[queies_df["text_len"] >= 10].to_dict(orient="records")
queies_dicts = queies_df.to_dict(orient="records")
shuffle(queies_dicts)

test_quantity = 1118

test_results = []
for num, d in enumerate(queies_dicts[:test_quantity]):
    print(num + 1, "/", test_quantity)
    # q = d["Query"]
    q_request = {"pubid": 9, "text": re.sub("\n", " ", d["Query"])}
    res = requests.post("http://0.0.0.0:8090/api/search", json=q_request)
    # res = requests.post("http://srv01.nlp.dev.msk2.sl.amedia.tech:4011/api/search", json=q_request)
    res_dict = {**d, **res.json()}
    test_results.append(res_dict)

test_results_df = pd.DataFrame(test_results)
print(test_results_df)
test_results_df.to_csv(os.path.join("results", "exp_supp_testing_240410.csv"), sep="\t", index=False)