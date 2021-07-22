import re
import collections


def remove_char(value):
    value = re.sub(r'[\[\],!.;#$^*_—<>/=%&?@"\'-:]', ' ', str(value))
    l_temp = [i for i in value.split()]
    return l_temp


def remove_dup(line):
    """ Remove duplicated words, first remove front ones. """
    l_temp = []

    i = len(line) - 1
    while i >= 0:
        line[i] = line[i].lower()
        if line[i] not in l_temp:
            l_temp.append(line[i])
        i = i - 1

    l_temp.reverse()
    return l_temp
    

def filter_words(line, count):
    """ Filter words in documents less than count. """
    cnt = collections.Counter()
    for sentence in line:
        cnt.update(sentence)

    s = set(word for word in cnt if cnt[word] < count)
    l_temp = [[word for word in sentence if word not in s] for sentence in line]
    return l_temp
