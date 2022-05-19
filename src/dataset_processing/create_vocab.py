import numpy as np
import os
import re
my_dict = {}
articles = 'C:\\Users\\jskag\\Datasets\\wikipedia\\en\\articles\\'
total_words = None
for art in os.listdir(articles):
    try:
        print(art)
        text = re.sub('[\W!@#$%^&*(){}\[\]:;"\',./]', ' ', open(articles + art, encoding="utf8").read().lower()).split()
        words = set(text)
        for word in words:
            try:
                if word in my_dict.keys():
                    my_dict[word] += 1
                else:
                    my_dict[word] = 1
            except Exception:
                pass
    except Exception:
        pass
f = open("vocab.csv", "a")
for word in my_dict:
    try:
        if my_dict[word] > 1000:
            f.write(word + ',')
    except Exception:
        pass
