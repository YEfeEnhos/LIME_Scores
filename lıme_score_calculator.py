import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics.pairwise import cosine_distances
from scipy.spatial.distance import hamming
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch

def cosine_distance(v1, v2):
    return cosine_distances([v1], [v2])[0][0]

def hamming_distance(sent1, sent2):
    return sum(c1 != c2 for c1, c2 in zip(sent1, sent2)) / max(len(sent1), len(sent2))

def compute_distance(original_sentence, adv_sentence, original_vector, adv_vector):
    cos_dist = cosine_distance(original_vector, adv_vector)
    ham_dist = hamming_distance(original_sentence, adv_sentence)
    return cos_dist * ham_dist

def compute_weight(distance, sigma=1.0):
    return np.exp(-distance / (2 * sigma ** 2))

def compute_LIME_score(original_sentence, adversarial_sentences, model, tokenizer, device='cpu'):
    original_encoded = tokenizer(original_sentence, return_tensors='pt', truncation=True, padding=True).to(device)
    with torch.no_grad():
        original_vector = model(**original_encoded).last_hidden_state.mean(dim=1).cpu().numpy()

    distances = []
    weights = []
    features = []

    for adv_sentence in adversarial_sentences:
        adv_encoded = tokenizer(adv_sentence, return_tensors='pt', truncation=True, padding=True).to(device)
        with torch.no_grad():
            adv_vector = model(**adv_encoded).last_hidden_state.mean(dim=1).cpu().numpy()

        distance = compute_distance(original_sentence, adv_sentence, original_vector, adv_vector)
        weight = compute_weight(distance)

        distances.append(distance)
        weights.append(weight)
        

        feature_vector = [1 if o != a else 0 for o, a in zip(original_sentence.split(), adv_sentence.split())]
        features.append(feature_vector)
    

    features = np.array(features)
    weights = np.array(weights)
    
    reg_model = LinearRegression()
    reg_model.fit(features, distances, sample_weight=weights)

    lime_scores = reg_model.coef_ * np.sign(reg_model.coef_)
    
    return lime_scores

# Example usage
if __name__ == "__main__":
    original_sentence = "Replace with your original sentence."
    adversarial_sentences = [
        "Replace with your adversarial sentence.",
        "",
        ""
    ]

    # Load your custom fine-tuned BERT model and tokenizer
    model_directory = "./fine_tuned_bert"  # Replace with the path to your custom model directory
    tokenizer = BertTokenizer.from_pretrained(model_directory)
    model = BertModel.from_pretrained(model_directory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    lime_scores = compute_LIME_score(original_sentence, adversarial_sentences, model, tokenizer, device=device)
    print("LIME Scores for each substitution:", lime_scores)
