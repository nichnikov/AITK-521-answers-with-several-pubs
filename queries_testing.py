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

test_quantity = 2688

test_results = []
for num, d in enumerate(queies_dicts[:test_quantity]):
    try:
        temp_results = {}
        print(num + 1, "/", test_quantity)
        for pubid in [9, 16, 17, 25]:
            # q = d["Query"]
            q_request = {"pubid": pubid, "text": re.sub("\n", " ", d["Query"])}
            res = requests.post("http://0.0.0.0:8090/api/search", json=q_request)
            res_d = res.json()
            res_d_with_pub = {}
            for name in ["templateId", "templateText", "sbert_best_id", "sbert_best_etalon", 
                        "sbert_best_answer", "sbert_score", "T5Opinion", "T5Score"]:
                res_d_with_pub["".join([name, "_", str(pubid)])] = res_d[name]
            # res = requests.post("http://srv01.nlp.dev.msk2.sl.amedia.tech:4011/api/search", json=q_request)
            temp_results = {**temp_results, **d, **res_d_with_pub}
        test_results.append(temp_results)
    except:
        pass

test_results_df = pd.DataFrame(test_results)

print(test_results_df)
test_results_df.to_csv(os.path.join("results", "exp_supp_testing_pubs240411.csv"), sep="\t", index=False)