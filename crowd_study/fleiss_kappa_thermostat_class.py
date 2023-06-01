import pandas as pd
from statsmodels.stats.inter_rater import aggregate_raters
import statsmodels.api as sm


class fleiss_kappa:
    def __init__(self):
        self.data_in: list[dict] = []
        self.data_between: list[list[tuple]] = []
        self.data_fleiss: list[list] = []
        self.words: list = []

    def doit(self, datasets, col_names, filter_words, header=0, filter_position=0, mode="fleiss"):
        """
        general method for fleiss kappa
        """
        self.load_data(datasets, col_names, header=header)
        self.onehot(filter_words, position=filter_position)
        self.remove_key()
        return self.fleiss_kappa(mode=mode)


    def load_data(self, dataset_path, csv_col_names, header=0):
        """
        loads csv files from path
        """
        for csv_path in dataset_path:
            data = pd.read_csv(csv_path, names=csv_col_names, header=header)
            if '_G' in csv_path:
                data.study = 'new'
            else:
                data.study = 'old'

            if '_GPT' in csv_path:
                data = data.dropna()

            data = self.pandas_to_dict(data)
            self.data_in.append(data)
        return self.data_in

    def onehot(self, filter_words, position):
        """
        encodes filterwords in data with there position,
        position: reference to position in pandas_to_dict
        """
        data = list(self.zip_dict(self.data_in[:]))
        data = self.one_hot_word(filter_words, data=data, position=position)
        self.data_between = data
        return data

    def remove_key(self):
        """
        removes first element in list, which was the dict key
        """
        for dat in range(len(self.data_between)):
            data = self.tuple_to_list(self.data_between[dat])
            data.pop(0)
            self.data_fleiss.append(data)
        return self.data_fleiss

    def fleiss_kappa(self, mode="fleiss"):
        """
        calculates fleiss kappa
        """
        ncat = len(self.data_fleiss)
        table = aggregate_raters(self.data_fleiss, ncat)[0]
        fleiss = sm.stats.fleiss_kappa(table, mode)
        return fleiss


    @classmethod
    def zip_dict(cls, *dicts):
        """
            zip function for dicts. converts dict[key] to tuple and concatenate them
            return: list(key,tuple,tuple,.....)
        """
        all_keys = set().union(*(d.keys() for d in dicts[0]))
        for i in all_keys:
        #for i in set(dicts[0][0]).intersection(*dicts[0][1:]):
            #yield list((i,) + tuple(d[i] for d in dicts[0]))
            yield [i] + [d[i] for d in dicts[0] if d.get(i) is not None]


    @classmethod
    def one_hot_word(cls, words, data: list, position=0):
        """
        takes list object, [ [key,tuple,tuple],[key,tuple,tuple],...]. return [ [key,OneHot(tuple,position),OneHot(tuple,position)], [key,OneHot(tuple,position),OneHot(tuple,position)],...]
        words: strings which will be OneHot by there index in words
        data: list like object containing tuples
        position: int, position of OneHot in tuples
        """
        amount_sub = len(data[0]) - 1
        for word in words:  # iter over all words
            for i in range(len(data)):  # iter over all keys
                for q in range(amount_sub):  # iter over all tuples per key
                    if data[i][q + 1][position] == word:  # replacement logic
                        a = list(data[i][q + 1])
                        indx = words.index(word)
                        a[position] = indx
                        data[i][q + 1] = list(a)
        return data

    @classmethod
    def tuple_to_list(cls, liste):
        """
        [int,tuple,tuple..]-> [int,int,int]
        """
        ls = []
        for i in range(len(liste)):
            try:
                ls.append(int(liste[i]))
            except:
                ls.append(int(list(liste[i])[0]))
        return ls

    @classmethod
    def pandas_to_dict(cls, pandasdataframe):
        """
        position in onehot is position here + 1
        """
        return dict(zip(list(pandasdataframe["idx"]),
                    zip(list(pandasdataframe["simulation"]),
                        list(pandasdataframe["helpful"]),
                        list(pandasdataframe["easy"]))))


