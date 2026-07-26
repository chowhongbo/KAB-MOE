from datasets import load_dataset
import os

GQA_RAW_IMAGE_DATASET = None
GQA_ID2IMAGE = None


def gqa_doc_to_visual(doc):
    global GQA_RAW_IMAGE_DATASET
    global GQA_ID2IMAGE
    if GQA_RAW_IMAGE_DATASET is None:
        gqa_image_root = os.getenv(
            "GQA_IMAGE_ROOT", os.path.join("datasets", "GQA/testdev_balanced_images")
        )
        GQA_RAW_IMAGE_DATASET = load_dataset(
            gqa_image_root, token=True
        )["train"]
        GQA_ID2IMAGE = {}
        for row in GQA_RAW_IMAGE_DATASET:
            GQA_ID2IMAGE[row["id"]] = row["image"].convert("RGB")
    image = GQA_ID2IMAGE[doc["imageId"]]
    return [image]


def gqa_doc_to_text(doc):
    question = doc["question"]
    return f"{question}\nAnswer the question using a single word or phrase."


def gqa_doc_to_org_text(doc):
    return doc["question"]


def gqa_doc_to_answer(doc):
    return doc["answer"]


def gqa_doc_to_full_answer(doc):
    return doc["fullAnswer"]


def gqa_transform(batch):
    # e.g., batch['imageId'] = ['id1', 'id2', ...]
    processed_texts = []
    processed_visuals = []
    processed_answers = []
    processed_full_answers = []
    processed_org_texts = []
    for img_id, q_text, a_text, f_a_text in zip(
        batch["imageId"], batch["question"], batch["answer"], batch["fullAnswer"]
    ):
        doc = {
            "imageId": img_id,
            "question": q_text,
            "answer": a_text,
            "fullAnswer": f_a_text,
        }
        visuals = gqa_doc_to_visual(doc)
        text = gqa_doc_to_text(doc)
        org_text = gqa_doc_to_org_text(doc)
        answer = gqa_doc_to_answer(doc)
        full_answer = gqa_doc_to_full_answer(doc)
        processed_visuals.append(visuals[0])
        processed_texts.append(text)
        processed_answers.append(answer)
        processed_full_answers.append(full_answer)
        processed_org_texts.append(org_text)
    return {
        "model_input_text": processed_texts,
        "model_input_visual": processed_visuals,
        "model_input_answer": processed_answers,
        "model_input_full_answer": processed_full_answers,
        "model_input_org_text": processed_org_texts,
    }
