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
    with open(file_path, 'r', encoding='utf-8') as f:
        text_content = f.read()

    sentences = sent_tokenize(text_content, language='russian')
    morph = pymorphy3.MorphAnalyzer()
    matched_pairs = set()
    
    for sentence in sentences:
        tokens = word_tokenize(sentence.lower(), language='russian')
        words = [t for t in tokens if re.match(r"^[а-яё-]+$", t)]
        
        for w1, w2 in zip(words, words[1:]):
            p1 = morph.parse(w1)[0]
            p2 = morph.parse(w2)[0]
            
            pos1, pos2 = p1.tag.POS, p2.tag.POS
            
            # Noun/ Adjective pairs
            if pos1 in {'NOUN', 'ADJF'} and pos2 in {'NOUN', 'ADJF'}:
                gender1, gender2 = p1.tag.gender, p2.tag.gender
                number1, number2 = p1.tag.number, p2.tag.number
                case1, case2 = p1.tag.case, p2.tag.case
                
                # Check for matching gender, number, and case
                if (gender1 and gender2 and gender1 == gender2 and
                    number1 and number2 and number1 == number2 and
                    case1 and case2 and case1 == case2):
                    pair = f"{p1.normal_form} {p2.normal_form}"
                    matched_pairs.add(pair)
    
    return sorted(matched_pairs)

def main():
    ensure_nltk_data()
    input_file = 'text.txt'
    results = process_text(input_file)

    if results:
        print("Найденные пары:\n")
        for pair in results:
            print(pair)
    else:
        print("Совпадающие пары не найдены.")

if __name__ == "__main__":
    main()