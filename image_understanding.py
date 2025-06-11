import os
from tqdm import tqdm
import torch
from transformers import Blip2Processor, Blip2Model, Blip2ForConditionalGeneration
import json
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Blip2Processor.from_pretrained("./res/Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("./res/Salesforce/blip2-flan-t5-xl", torch_dtype=torch.float16)
model.to(device)


def read_json_data(file_name):
    with open(file_name) as f:
        article_list = [json.loads(line) for line in f]
        return article_list


def blip2_captioning_inference(image):
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    generated_ids = model.generate(**inputs)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text


def image_understanding():
    test_samples = read_json_data(os.path.join('res', "cosmos_anns_acm", "acm_anns", 'public_test_acm.json'))
    responses = []

    for index, v_data in enumerate(tqdm(test_samples)):
        torch.cuda.empty_cache()

        img_path = os.path.join('res', v_data["img_local_path"])
        raw_image = Image.open(img_path).convert("RGB")
        response = blip2_captioning_inference(raw_image)
        responses.append(response + "\n")

    with open('res/Blip2_responses/Captioning.txt', 'w') as f:
        f.writelines(responses)


def main():
    # 对测试集图像生成文字描述，保存成文件
    image_understanding()


if __name__ == "__main__":
    main()