import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

class TextSimilarity():

    def __init__(self, model_name = 'bert-base-chinese'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def tokenize_sentences(self, sentences):
        tokens = {'input_ids': [], "attention_mask" : []}
        for sentence in sentences:
            new_tokens = tokenizer.encode_plus(sentence, max_length=512, truncation=True, padding='max_length', return_tensors = 'pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens["attention_mask"].append(new_tokens["attention_mask"][0])
        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens["attention_mask"] = torch.stack(tokens["attention_mask"])
        tokens['input_ids'].shape
        outputs = self.model(**tokens)
        embeddings = outputs.last_hidden_state
        attention = tokens['attention_mask']
        mask = attention.unsqueeze(-1).expand(embeddings.shape).float()
        mask_embeddings = mask * embeddings
        summed = torch.sum(mask_embeddings, 1)
        counts = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = summed/counts

        return mean_pooled
    
    def cosine_sim(mean_pooled, indexOne, indexTwo):
        return cosine_similarity([mean_pooled[indexOne]], mean_pooled[indexTwo])
    
    




