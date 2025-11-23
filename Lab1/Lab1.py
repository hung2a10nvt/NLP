import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import pymorphy3
import re

def ensure_nltk_data():
    for package in ['punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{package}')
        except LookupError:
            nltk.download(package)

def process_text(file_path):    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text_content = f.read()
    except FileNotFoundError:
        print(f"Error: Can't find {file_path}")
        return []

    sentences = sent_tokenize(text_content, language='russian')
    morph = pymorphy3.MorphAnalyzer()
    matched_pairs = set()
    
    TARGET_POS = {'NOUN', 'ADJF', 'PRTF'} 

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower(), language='russian')
        words = [t for t in tokens if re.match(r"^[а-яё-]+$", t)]
        
        for w1, w2 in zip(words, words[1:]):
            p1 = morph.parse(w1)[0]
            p2 = morph.parse(w2)[0]
            
            tag1, tag2 = p1.tag, p2.tag
            
            # Check if is Noun/Adj/Participle
            if tag1.POS in TARGET_POS and tag2.POS in TARGET_POS:
                
                # Number
                if tag1.number != tag2.number or not tag1.number:
                    continue
                
                # Case
                if tag1.case != tag2.case or not tag1.case:
                    continue
                
                # Gender
                # if adj's number = plur, pymorphy returns gender=None 
                if tag1.number == 'sing':
                    if tag1.gender != tag2.gender or not tag1.gender:
                        continue
                
                pair = f"{p1.normal_form} {p2.normal_form}"
                matched_pairs.add(pair)
    
    return sorted(list(matched_pairs))

def main():
    ensure_nltk_data()
    input_file = 'text.txt'
    results = process_text(input_file)

    if results:
        print(f"Found {len(results)} pairs:\n")
        for pair in results:
            print(pair)
    else:
        print("No pair found.")

if __name__ == "__main__":
    main()