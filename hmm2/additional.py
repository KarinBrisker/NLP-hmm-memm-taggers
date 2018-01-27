import numpy as np
import sys

class QModel:

    def __init__(self):
        self.q_1_dict = {}
        self.q_2_dict = {}
        self.q_3_dict = {}
        self.Q_dict = {}
        self.total_word_count = 0
        self.l_1 = 0.8
        self.l_2 = 0.15
        self.l_3 = 0.05

    def load_from_file(self, file_name):
        for line in file(file_name):
            tags_count = line.split('\t')
            tags = tags_count[0].split(' ')
            count = int(tags_count[1])
            if tags.__len__() == 1:
                self.q_1_dict[tags[0]] = count
                self.total_word_count += count
            elif tags.__len__() == 2:
                self.q_2_dict[tags_count[0]] = count
            else:
                self.q_3_dict[tags_count[0]] = count
        for a in self.q_1_dict.keys():
            for b in self.q_1_dict.keys():
                for c in self.q_1_dict.keys():
                    self.Q_dict[a + ' ' + b + ' ' + c] = self.get_q(c, b, a)

    def get_q(self, c, b, a):
        count_c = self.q_1_dict[c]
        count_bc = self.q_2_dict.get(b + ' ' + c, 0)
        count_abc = self.q_3_dict.get(a + ' ' + b + ' ' + c, 0)
        count_b = self.q_1_dict[b]
        count_ab = self.q_2_dict.get(a + ' ' + b, 0)
        q_c_if_ab = 0
        if count_abc != 0:
            q_c_if_ab = float(count_abc) / count_ab
        q_c_if_b = 0
        if count_bc != 0:
            q_c_if_b = float(count_bc) / count_b
        q_c = float(count_c) / self.total_word_count
        return self.l_1 * q_c_if_ab + self.l_2 * q_c_if_b + self.l_3 * q_c

    def count_tag(self, tag):
        return self.q_1_dict[tag]

    def get_Q(self, c, b, a):
        return self.Q_dict[a + ' ' + b + ' ' + c]

class EModel:

    def __init__(self):
        self.e_dict = {}
        self.unk_classes = ['UNK-ED', 'UNK-ING', 'UN-UNK', 'UNK-AL', 'UNK-NUMBER',
                            'UNK-ANCE', 'UNK-TION', 'UNK-SION', 'UNK-URE',
                            'UNK-MENT', 'UNK-AGE', 'UNK-ERY', 'UNK-IVE', 'UNK-FUL',
                            'UNK-ABLE', 'UNK-DOM', 'UNK-EN', 'UNK-ISM', 'UNK-ISH',
                            'UNK-ER', 'UNK-SHIP', 'UNK-NESS', 'UNK-FY', 'UNK-IZE',
                            'A-UNK', 'ANTI-UNK', 'EX-UNK', 'I-UNK','IN-UNK','UNK-TIME',
                            'UNK-ln-4', 'UNK-ln-9', 'UNK-ln-12', 'UNK-LONG']
        for clss in self.unk_classes:
            self.e_dict[clss] = {}

    def load_from_file(self, file_name):
        unique_unk_class_counter = {}
        for line in file(file_name):
            word_count = line.split('\t')
            count = int(word_count[1])
            word_tag = word_count[0].split(' ')
            word = word_tag[0]
            tag = word_tag[1]
            if word not in self.e_dict:
                self.e_dict[word] = {}
            self.e_dict[word][tag] = count

            unk_class = self.classify_unknown(word)
            if tag not in self.e_dict[unk_class]:
                self.e_dict[unk_class][tag] = count
            else:
                self.e_dict[unk_class][tag] += count
            if unk_class + ' ' + tag not in unique_unk_class_counter:
                unique_unk_class_counter[unk_class + ' ' + tag] = 1
            else:
                unique_unk_class_counter[unk_class + ' ' + tag] += 1
        # keys_to_delete = []
        # for word in self.e_dict:
        #     if len(self.e_dict[word]) == 1:
        #         if self.e_dict[word][self.e_dict[word].keys()[0]] < 1:
        #             keys_to_delete.append(word)
        # for word in keys_to_delete:
        #     del self.e_dict[word]
        for clss in self.unk_classes:
            for tag in self.e_dict[clss]:
                self.e_dict[clss][tag] /= unique_unk_class_counter[clss + ' ' + tag]

    def num_there(self, s):
        return any(i.isdigit() for i in s)

    def classify_unknown(self, word):
        if word.replace('.', '').replace(',', '').isdigit():
            return 'UNK-NUMBER'
        if word.replace(':', '', 1).isdigit():
            return 'UNK-TIME'
        if len(word) >= 7:
            if word[-3:] == 'ing':
                return 'UNK-ING'
            if word[-2:] == 'ed':
                return 'UNK-ED'
            if word[:4] == 'Anti' or word[:4] == 'anti':
                return 'ANTI-UNK'
            if word[:1] == 'A' or word[:1] == 'a':
                return 'A-UNK'
            if word[:1] == 'Ex' or word[:1] == 'ex':
                return 'EX-UNK'
            if word[:1] == 'In' or word[:1] == 'in':
                return 'IN-UNK'
            if word[:1] == 'I' or word[:1] == 'i':
                return 'I-UNK'
            if word[-4:] == 'able':
                return 'UNK-ABLE'
            if word[-4:] == 'ance' or word[-4:] == 'ence':
                return 'UNK-ANCE'
            if word[-4:] == 'tion':
                return 'UNK-TION'
            if word[-4:] == 'sion':
                return 'UNK-SION'
            if word[-4:] == 'ship':
                return 'UNK-SHIP'
            if word[-4:] == 'ness':
                return 'UNK-NESS'
            if word[-4:] == 'ment':
                return 'UNK-MENT'
            if word[-2:] == 'al':
                return 'UNK-AL'
            if word[-3:] == 'dom':
                return 'UNK-DOM'
            if word[-3:] == 'ure':
                return 'UNK-URE'
            if word[-3:] == 'age':
                return 'UNK-AGE'
            if word[-3:] == 'ish':
                return 'UNK-ISH'
            if word[-3:] == 'ful':
                return 'UNK-FUL'
            if word[-3:] == 'ive':
                return 'UNK-IVE'
            if word[-3:] == 'ism':
                return 'UNK-ISM'
            if word[-3:] == 'ize':
                return 'UNK-IZE'
            if word[-2:] == 'er':
                return 'UNK-ER'
            if word[-2:] == 'en':
                return 'UNK-EN'
            if word[-2:] == 'fy':
                return 'UNK-FY'
            if word[:2] == 'un' or word[:2] == 'Un':
                return 'UN-UNK'
        if len(word) <= 5:
            return 'UNK-ln-4'
        elif len(word) <= 8:
            return 'UNK-ln-9'
        elif len(word) <= 12:
            return 'UNK-ln-12'
        return 'UNK-LONG'

    def get_tags_for(self, word):
        if not word in self.e_dict:
            word = self.classify_unknown(word)
        return self.e_dict[word].keys()

    def get_e(self, word, tag, q_model):
        if word not in self.e_dict:
            word = self.classify_unknown(word)
        return float(self.e_dict[word][tag]) / q_model.count_tag(tag)


class HMMSCORE:

    def __init__(self, q_m, e_m):
        self.qmodel = q_m
        self.emodel = e_m

    def getScore(self, word,tag,prev,prev_prev):
        e = self.emodel.get_e(word, tag, self.qmodel)
        q = self.qmodel.get_Q(tag,prev,prev_prev)
        return np.log(e) + np.log(q)


def read_data(file_name):
    data = []
    for line in file(file_name):
        words_tags = line.strip().split(" ")
        data.append([i.rsplit("/", 1) for i in words_tags])
    return data


def calc_accuracy(output_set, truth_set):
    good = 0
    bad = 0
    for i in range(len(output_set)):
        for j in range(len(output_set[i])):
            tagTrue = truth_set[i][j][1]
            tagMy = output_set[i][j][1]
            if tagTrue == tagMy:
                good += 1
            else:
                bad += 1
    return good / float(good + bad)


def create_e_dictionary(dset):
    e_dict = {}
    for line in dset:
        for word_tag in line:
            word = word_tag[0]
            tag = word_tag[1]
            if word in e_dict:
                if tag in e_dict[word]:
                    e_dict[word][tag] += 1
                else:
                    e_dict[word][tag] = 1
            else:
                e_dict[word] = {}
                e_dict[word][tag] = 1
    return e_dict


def create_q_dictionary(dset):
    q_3_dict = {}
    q_2_dict = {}
    q_1_dict = {}

    q_1_dict['START'] = len(dset)
    q_2_dict['START START'] = len(dset)

    for line in dset:
        n = len(line)
        for i in range(0, n):
            t = line[i][1]
            if q_1_dict.has_key(t):
                q_1_dict[t] += 1
            else:
                q_1_dict[t] = 1

        for i in range(0, n):
            if i > 0:
                t1 = line[i - 1][1]
            else:
                t1 = 'START'
            t2 = line[i][1]
            t = t1 + ' ' + t2
            if q_2_dict.has_key(t):
                q_2_dict[t] += 1
            else:
                q_2_dict[t] = 1

        for i in range(0, n):
            if i > 1:
                t1 = line[i - 2][1]
            else:
                t1 = 'START'
            if i > 0:
                t2 = line[i - 1][1]
            else:
                t2 = 'START'
            t3 = line[i][1]
            t = t1 + ' ' + t2 + ' ' + t3
            if q_3_dict.has_key(t):
                q_3_dict[t] += 1
            else:
                q_3_dict[t] = 1
    return q_1_dict, q_2_dict, q_3_dict


def save_e_to_file(e_dict, file_name):
    file_e = open(file_name, 'w')
    for word in e_dict.keys():
        for tag in e_dict[word].keys():
            file_e.write(word + ' ' + tag + '\t' + str(e_dict[word][tag]) + '\n')
    file_e.close()


def save_q_to_file(q_dicts, file_name):
    file_q = open(file_name, 'w')
    for q in q_dicts[0].keys():
        file_q.write(q + '\t' + str(q_dicts[0][q]) + '\n')
    for q in q_dicts[1].keys():
        file_q.write(q + '\t' + str(q_dicts[1][q]) + '\n')
    for q in q_dicts[2].keys():
        file_q.write(q + '\t' + str(q_dicts[2][q]) + '\n')
    file_q.close()


if __name__ == '__main__':
    # read the program arguments
    input_file_name = sys.argv[1]
    q_mle_filename = sys.argv[2]
    e_mle_filename = sys.argv[3]

    train_set = read_data(input_file_name)
    e_dictionary = create_e_dictionary(train_set)
    save_e_to_file(e_dictionary, e_mle_filename)
    print 'saved e to file'
    q_dictionaries = create_q_dictionary(train_set)
    save_q_to_file(q_dictionaries, q_mle_filename)
    print 'saved q to file'