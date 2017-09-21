import collections
import csv
import re
from pymystem3 import Mystem

m = Mystem()
keyword = 'Apple'
counter = collections.Counter()
stopwords = ["большой", "бы"", ""быть", "в", "во", "за", "с", "весь", "вот", "все", "всей", "вы", "говорить",
             "год", "да", "для", "до", "еще", "же", "знать", "и", "или", "из", "к", "как", "который",
             "мочь", "мы", "на", "наш", "не", "него", "нее", "нет", "них", "но", "о", "один",
             "она", "они", "оно", "оный", "от", "ото", "по", "с", "свой", "себя", "сказть", "такой",
             "только", "тот", "ты", "у", "что", "это", "этот", "я", "под", "уже"]

def find_bigrams(corpus):
    bigram_list = []
    with open(corpus, 'r', encoding='utf-8') as f:
        text = f.readlines()
    for line in text:
        line_split = line.split(' ')
        for i in range(len(line_split)-1):
            if line_split[i] == keyword:
                word2 = m.lemmatize(line_split[i+1])
                counter[line_split[i]] += 1
                #counter[word2[0]] += 1
                counter[(line_split[i], word2[0])] += 1
                if word2[0] not in bigram_list:
                    bigram_list.append(word2[0])
            if m.lemmatize(line_split[i]) in bigram_list:
                counter[line_split[i]] +=1
    with open('bigrams.csv', 'a', encoding = 'utf-8', newline='') as csvfile:
        fieldnames = ['word1', 'word2', 'count(word1)', 'count(word2)', 'count(bigram)']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter = '\t')
        writer.writeheader()
        t = 0
        while t < len(set(bigram_list)):
            if counter[("Apple", bigram_list[t])] >= 3 and bigram_list[t] not in stopwords:
                writer.writerow({'word1': "Apple", 'word2': bigram_list[t], 'count(word1)': counter["Apple"], 'count(word2)': counter[bigram_list[t]], 'count(bigram)': counter[("Apple", bigram_list[t])]})
            t += 1

    return counter

def main():
    corpus = 'rus-wikipedia-sample-companies.txt'
    find_bigrams(corpus)
    #print(find_bigrams(corpus))

if __name__ == '__main__':
    main()