

from transformers import AutoTokenizer, AutoModel
import torch

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[
        0
    ]  # First element of model_output contains all token embeddings
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


from sentence_transformers import SentenceTransformer
# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained("hiiamsid/sentence_similarity_spanish_es")
model = AutoModel.from_pretrained("hiiamsid/sentence_similarity_spanish_es")

def get_tokenizer():
    return tokenizer

def text_to_vector(text):
    # Tokenize sentences
    encoded_input = tokenizer([text], padding=True, truncation=True, return_tensors="pt")
    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)
    # Perform pooling. In this case, max pooling.
    sentence_embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    #   print(sentence_embeddings)
    return sentence_embeddings.cpu().numpy()


