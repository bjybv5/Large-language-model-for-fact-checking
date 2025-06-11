import time
import os
import numpy as np
from tqdm import tqdm
import json
import re
import openai
import warnings


warnings.filterwarnings("ignore")

openai.api_key = os.getenv("OPENAI_API_KEY")


def simple_chat(prompt):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    output_text = completion.choices[0].message["content"]

    return output_text


def read_json_data(file_name):
    with open(file_name) as f:
        article_list = [json.loads(line) for line in f]
        return article_list


def appnend_sign(caption):
    if caption[-1] != "." and caption[-1] != "\'"\
            and caption[-1] != " " and caption[-1] != "\"" and caption[-1] != "”"\
            and caption[-1] != "’" and caption[-1] != " " and caption[-1] != "!"\
            and caption[-1] != "?" and caption[-1] != "," and caption[-1] != ":":
        caption = caption + "."
    if caption[-1] == " ":
        if caption[-2] != "." and caption[-2] != "\n":
            caption = caption + "."
    return caption


def sentence_transform(caption):
    return "The image is talking about " + caption


def generate_prompt_contradictory(cap1, cap2):
    cap1 = appnend_sign(cap1)
    cap2 = appnend_sign(cap2)
    prompt = "You are a detective now, I will tell two propositions and you will respond with whether the two propositions are contradictory.\nProposition A: " + cap1 + "\nProposition B: " + cap2

    return prompt


def generate_prompt_semantics(cap1, cap2):
    cap1 = appnend_sign(cap1)
    cap2 = appnend_sign(cap2)
    prompt = "You are a detective now. I will tell two propositions and you will respond with whether two propositions have the same semantics.\nProposition A: {}\nProposition B: {}".format(cap1, cap2)

    return prompt


def generate_prompt_conflict(capi, cap1, cap2):
    capi = appnend_sign(capi)
    cap1 = appnend_sign(cap1)
    cap2 = appnend_sign(cap2)

    cap1 = sentence_transform(cap1)
    cap2 = sentence_transform(cap2)
    propmt ="I tell you the content of an image and two paragraphs of text. These two pieces of text are used to explain the same image, and you have to judge whether the two interpretations conflict. "\
            + " Please answer in the following json format only. {\"Conflict\": \"<Yes/No>\", \"Explanation\": \"<>\"}\nDescription of Image Content: " + capi + "\nText 1: " + cap1 + "\nText 2: " + cap2

    return propmt


def construct_responses_contradictory():
    test_samples = read_json_data(os.path.join("res", "cosmos_anns_acm", "acm_anns", 'public_test_acm.json'))
    responses = []

    print(generate_prompt_semantics("cap1", "cap2"))

    for index, v_data in enumerate(tqdm(test_samples)):
        cap1 = v_data['caption1']
        cap2 = v_data['caption2']

        prompt = generate_prompt_contradictory(cap1, cap2)

        flag = 0
        while flag == 0:
            try:
                output_text = simple_chat(prompt)
                flag = 1
            except:
                time.sleep(10)

        responses.append(output_text)

    with open('res/responses/Contradictory_C1_C2.json', 'w') as f:
        json.dump(responses, f)


def construct_responses_semantics():
    test_samples = read_json_data(os.path.join("res", "cosmos_anns_acm", "acm_anns", 'public_test_acm.json'))
    responses = []

    print(generate_prompt_semantics("cap1", "cap2"))

    for index, v_data in enumerate(tqdm(test_samples)):
        cap1 = v_data['caption1']
        cap2 = v_data['caption2']

        prompt = generate_prompt_semantics(cap1, cap2)

        flag = 0
        while flag == 0:
            try:
                output_text = simple_chat(prompt)
                flag = 1
            except:
                time.sleep(10)

        responses.append(output_text)

    with open('res/responses/Semantics_C1_C2.json', 'w') as f:
        json.dump(responses, f)


def construct_responses_conflict():
    with open('res/Blip2_responses/Captioning.txt', 'r') as f:
        captionings = f.readlines()

    test_samples = read_json_data(os.path.join("res", "cosmos_anns_acm", "acm_anns", 'public_test_acm.json'))
    responses = []

    print(generate_prompt_conflict("capi", "cap1", "cap2"))

    for index, v_data in enumerate(tqdm(test_samples)):
        capi = captionings[index]
        cap1 = v_data['caption1']
        cap2 = v_data['caption2']

        prompt = generate_prompt_conflict(capi, cap1, cap2)

        flag = 0
        while flag == 0:
            try:
                output_text = simple_chat(prompt)
                flag = 1
            except:
                time.sleep(10)

        responses.append(output_text)

    with open('res/responses/Conflict_P11_T0.json', 'w') as f:
        json.dump(responses, f)


def read_contradictory_predicts():
    prompt = generate_prompt_contradictory("Caption1", "Caption2")
    # print(prompt)

    predicts = []
    count_yes = 0
    count_no = 0

    with open('res/responses/Contradictory_C1_C2.json', 'r') as f:
        data = json.load(f)
        for index, response in enumerate(data):
            result = 1

            if "contradictory" in response or "contradict" in response \
                    or "contradiction" in response:
                vlist = re.split('[:;,."\s]\s*', response)
                # print(vlist)
                if "Yes" in vlist:
                    result = 1
                    count_yes += 1
                elif "not" in vlist or "Not" in vlist or "NOT" in vlist\
                        or "no" in vlist:
                    result = 0
                    count_no += 1
                else:
                    result = 1
                    count_yes += 1
            predicts.append(result)
    return predicts


def read_semantics_predicts():
    prompt = generate_prompt_semantics("Caption1", "Caption2")
    # print(prompt)

    predicts = []
    count_yes = 0
    count_no = 0

    with open('res/responses/Semantics_C1_C2.json', 'r') as f:
        data = json.load(f)
        for index, response in enumerate(data):
            vlist = re.split('[:;,."\s]\s*', response)
            if "Yes" in vlist:
                result = 0
                count_yes += 1
            elif "No" in vlist or "not" in vlist or "different" in vlist\
                    or "opposite" in vlist:
                result = 1
                count_no += 1
            else:
                result = 0
                count_yes += 1
            predicts.append(result)
    return predicts


def read_conflict_predicts():
    predicts = []
    count_yes = 0
    count_no = 0

    with open('res/responses/Conflict_P11_T0.json', 'r') as f:
        data = json.load(f)
        for index, sample in enumerate(data):
            response = json.loads(sample)
            # vlist = re.split('[:;,."\s]\s*', response["Conflict"])
            if response["Conflict"] == "Yes":
                result = 1
                count_yes += 1
            else:
                result = 0
                count_no += 1
            predicts.append(result)
    return predicts


def get_confusion_matrix(preds, labels):

    def flip_diag(a):
        w = np.einsum('ii->i', a)
        w[:] = w[::-1]
        return a

    conf_matrix = np.zeros((2, 2), dtype=int)
    for p, t in zip(preds, labels):
        conf_matrix[p, t] += 1
    return flip_diag(conf_matrix).T


def evaluate_responses():
    contradictory_predicts = read_contradictory_predicts()     # 判断两段文本是否矛盾

    semantics_predicts = read_semantics_predicts()   # 判断两段文本是否一致

    conflict_predicts = read_conflict_predicts()

    test_samples = read_json_data(os.path.join("res", "cosmos_anns_acm", "acm_anns", 'public_test_acm.json'))
    ours_correct = 0
    lang_correct = 0
    textual_sim_threshold = 0.5
    preds = []
    labels = []
    ids = []

    for index, v_data in enumerate(tqdm(test_samples)):
        actual_context = int(v_data['context_label'])
        language_context = 0 if float(v_data['bert_base_score']) >= textual_sim_threshold else 1

        if contradictory_predicts[index] == 1 and semantics_predicts[index] == 1:
            pred_context = 1
        if contradictory_predicts[index] == 0 and semantics_predicts[index] == 0:
            pred_context = 0
        if contradictory_predicts[index] == 1 and semantics_predicts[index] == 0:
            pred_context = conflict_predicts[index]
        if contradictory_predicts[index] == 0 and semantics_predicts[index] == 1:
            pred_context = conflict_predicts[index]

        preds.append(pred_context)
        labels.append(actual_context)

        if pred_context == actual_context:
            ours_correct += 1
        else:
            ids.append(index)

        if language_context == actual_context:
            lang_correct += 1

    print("ChatGPT Accuracy", ours_correct / len(test_samples))
    print("Language Baseline Accuracy", lang_correct / len(test_samples))

    print("Confusion Matrix:")
    print(get_confusion_matrix(preds, labels))


def main():
    # 调用大模型进行问答，保存成文件
    construct_responses_contradictory()
    construct_responses_semantics()
    construct_responses_conflict()

    # 读取问答结果文件，评估测试集
    evaluate_responses()


if __name__ == "__main__":
    main()