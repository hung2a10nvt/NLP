import gensim
import re

TARGET_WORDS = ["подход", "задача"]

pos = ["метод_NOUN", "проблема_NOUN"]
neg = []

MODEL_PATH = "cbow.txt" 

try:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
except FileNotFoundError:
    print("Ошибка: Файл модели не найден.")
    exit()

dist = word2vec.most_similar(positive=pos, negative=neg, topn=10)

pat = re.compile("(.*)_NOUN")

for i in dist:
    print(i)
for i in dist:
    e = pat.match(i[0])
    if e is not None:
        print(e.group(1))