# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:12:55 2019

@author: izumi
"""
import numpy as np
np.set_printoptions(linewidth=200, suppress=True, precision=3)
from plugins import consts
from plugins import models
from plugins import functions

from gensim.models import word2vec
import chainer.functions as F

"""
好きな発話から推定する．
"""

class Classify:
    def __init__(self, context):
        print('Context : '+str(context))
        self.context = int(context)
        self.models_bottom, self.models_top, self.models_ova, self.models_enova = functions.load_models(self.context)

    def classify(self, texts, w2v):
        print('Context length : ' + str(self.context))
        # print(texts)
        print()

        xs = functions.to_variable(texts, w2v)
        answers = []

        if xs != []:
            # Bottom
            ys_bottom, ys_bottom_max = [], []
            hs_bottom = []
            for model_bottom in self.models_bottom:
                hs,ys = model_bottom(xs)
                hs_bottom.append(hs)
                ys_bottom.append(F.softmax(ys).data[len(xs)-1])
                ys_bottom_max.append(np.argmax(F.softmax(ys).data[len(xs)-1]))
            
            print('Bottom :')
            bottom_ans = functions.print_answers(ys_bottom, ys_bottom_max)

            # Others
            ys_top, ys_top_max = [], []
            ys_ova, ys_ova_max = [], []
            ys_enova, ys_enova_max = [], []
            for top, ova, enova, xs in zip(self.models_top, self.models_ova, self.models_enova, hs_bottom):
                # Top
                y_top = F.softmax(top([xs])).data
                ys_top.append(y_top)
                ys_top_max.append(np.argmax(y_top))

                # OVA
                y_ova = F.softmax(ova([xs])).data
                ys_ova.append(y_ova)
                ys_ova_max.append(np.argmax(y_ova))

                # ENOVA
                y_enova = F.softmax(enova([xs])).data
                ys_enova.append(y_enova)
                ys_enova_max.append(np.argmax(y_enova))

            print('Top :')
            top_ans = functions.print_answers(ys_top, ys_top_max)

            print('OVA :')
            ova_ans = functions.print_answers(ys_ova, ys_ova_max)

            print('ENOVA :')
            enova_ans = functions.print_answers(ys_enova, ys_enova_max)

            answers.append(consts.ACTS[bottom_ans])
            answers.append(consts.ACTS[top_ans])
            answers.append(consts.ACTS[ova_ans])
            answers.append(consts.ACTS[enova_ans])

        return answers

    
if __name__ == '__main__':
    functions.reset_seed()
    texts = ['おはようございます','よろしくね','お元気ですか','へえー。']

    w2v = functions.load_w2v(consts.W2V_PATH)
    classify = Classify(len(texts))
    answers = classify.classify(texts, w2v)
    print('Answers:')
    print(answers)
    print()