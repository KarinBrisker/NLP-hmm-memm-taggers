import sys


def read_data(file_name):
    data = []
    for line in file(file_name):
        words_tags = line.strip().split(" ")
        data.append([i.rsplit("/", 1) for i in words_tags])
    return data


def turnWordToFeatures(wordparams, is_word_unique):
    word, word_p, word_pp, word_n, word_nn, tag_p, tag_pp = wordparams

    feature_list = []
    if not is_word_unique:
        feature_list.append('w=' + word)
    feature_list.append('w_p=' + word_p)
    feature_list.append('w_pp=' + word_pp)
    feature_list.append('w_n=' + word_n)
    feature_list.append('w_nn=' + word_nn)
    feature_list.append('t_p=' + tag_p)
    feature_list.append('t_pp=' + tag_pp + '_' + tag_p)

    if is_word_unique:
        feature_list.append('prf=' + word[:1].lower())
        feature_list.append('sfx=' + word[-1:].lower())
        if len(word) > 1:
            feature_list.append('prf=' + word[:2].lower())
            feature_list.append('sfx=' + word[-2:].lower())
        if len(word) > 2:
            feature_list.append('prf=' + word[:3].lower())
            feature_list.append('sfx=' + word[-3:].lower())
        if len(word) > 3:
            feature_list.append('prf=' + word[:4].lower())
            feature_list.append('sfx=' + word[-4:].lower())
        if any(i.isdigit() for i in word):
            feature_list.append('cnt_num')
        if any(i.isupper() for i in word):
            feature_list.append('cnt_upp')
        if any(i == '-' for i in word):
            feature_list.append('cnt_hyph')
    return  feature_list

if __name__ == '__main__':
    # read the program arguments
    corpus_file = sys.argv[1]
    features_file = sys.argv[2]
    word_tags = {}
    train_set = read_data(corpus_file)
    word_count_dict = {}
    for line in train_set:
        for word_tag in line:
            word = word_tag[0]
            tag = word_tag[1]
            if word not in word_count_dict:
                word_count_dict[word] = 1
                word_tags[word] = []
                word_tags[word].append(tag)
            else:
                if tag not in word_tags[word]:
                    word_tags[word].append(tag)
                word_count_dict[word] += 1

    file_feat = open(features_file, 'w')
    for line in train_set:
        line.append(['*n*', None])
        line.append(['*n*', None])
        line.insert(0, ['*n*', 'STRT'])
        line.insert(0, ['*n*', 'STRT'])
        for i in range(2, len(line) - 2):
            word = line[i][0]
            is_word_unique = word_count_dict[word] <= 4
            word_prev = line[i - 1][0]
            tag_prev = line[i - 1][1]
            word_prev_prev = line[i - 2][0]
            tag_prev_prev = line[i - 2][1]
            word_next = line[i + 1][0]
            word_next_next = line[i + 2][0]
            params = [word, word_prev, word_prev_prev, word_next, word_next_next, tag_prev, tag_prev_prev]

            true_tag = line[i][1]
            feature_str = true_tag + ' ' + ' '.join(turnWordToFeatures(params, is_word_unique))

            file_feat.write(feature_str + '\n')
    # write_word_tags_to_file(word_tags)
    file_feat.close()
