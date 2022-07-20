
from util import *
from ICLLDA import ICLLDA


if __name__ == '__main__':

    conf = ModelConf('ICLLDA.conf')
    for i in range(5):
        training_data = load_data_set(conf['training.set'] + "train_" + str(i) + ".txt")
        test_data = load_data_set(conf['test.set'] + "test_" + str(i) + ".txt")
        ICLLDA(conf=conf, training_set=training_data, test_set=test_data, i=i)
        ICLLDA.train()
