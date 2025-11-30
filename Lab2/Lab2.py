import gensim
import re
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

TARGET_WORDS = ["подход", "задача"]

pos = ["метод_NOUN", "проблема_NOUN"]
neg = []

MODEL_PATH = "cbow.txt" 

try:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
except FileNotFoundError:
    print("Ошибка: Файл модели не найден.")
    exit()

# topn=30, чтобы иметь запас, так как нам нужны только СУЩЕСТВИТЕЛЬНЫЕ
dist = word2vec.most_similar(positive=pos, negative=neg, topn=30)

pat = re.compile("(.*)_NOUN")

print(f"\nРезультат вычисления для: {' + '.join(pos)}")
print(f"Ожидаемые цели: {', '.join(TARGET_WORDS)}")

count = 0
found_targets = []

for word_tag, score in dist:
    # Проверка на соответствие шаблону _NOUN
    match = pat.match(word_tag)
    
    if match:
        clean_word = match.group(1) # Извлекаем чистое слово
        
        # Пропускаем, если слово совпадает с исходными
        if word_tag in pos:
            continue
            
        print(f"{count + 1}. {clean_word} (сходство: {score:.4f})")
        
        # Проверяем, нашли ли целевые слова
        if clean_word in TARGET_WORDS:
            found_targets.append(clean_word)
            
        count += 1
        
        # когда нашли 10 существительных
        if count == 10:
            break