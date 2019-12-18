from __future__ import unicode_literals
import re
import unicodedata
import numpy as np
import random
import copy
import chainer
from chainer import Chain,cuda,Variable,serializers
from gensim.models import word2vec
import os
import MeCab
from plugins import models
from plugins import consts

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    if cuda.available:
        cuda.cupy.random.seed(seed)


# 文書を正規化する関数群
def unicode_normalize(cls, s):
    pt = re.compile('([{}]+)'.format(cls))

    def norm(c):
        return unicodedata.normalize('NFKC', c) if pt.match(c) else c

    s = ''.join(norm(x) for x in re.split(pt, s))
    s = re.sub('－', '-', s)
    return s

def remove_extra_spaces(s):
    s = re.sub('[ 　]+', ' ', s)
    blocks = ''.join(('\u4E00-\u9FFF',  # CJK UNIFIED IDEOGRAPHS
                      '\u3040-\u309F',  # HIRAGANA
                      '\u30A0-\u30FF',  # KATAKANA
                      '\u3000-\u303F',  # CJK SYMBOLS AND PUNCTUATION
                      '\uFF00-\uFFEF'   # HALFWIDTH AND FULLWIDTH FORMS
                      ))
    basic_latin = '\u0000-\u007F'

    def remove_space_between(cls1, cls2, s):
        p = re.compile('([{}]) ([{}])'.format(cls1, cls2))
        while p.search(s):
            s = p.sub(r'\1\2', s)
        return s

    s = remove_space_between(blocks, blocks, s)
    s = remove_space_between(blocks, basic_latin, s)
    s = remove_space_between(basic_latin, blocks, s)
    return s

# 文章正規化
def normalize_neologd(s):
    s = s.strip()
    s = unicode_normalize('０-９Ａ-Ｚａ-ｚ｡-ﾟ', s)

    def maketrans(f, t):
        return {ord(x): ord(y) for x, y in zip(f, t)}

    s = re.sub('[˗֊‐‑‒–⁃⁻₋−]+', '-', s)  # normalize hyphens
    s = re.sub('[﹣－ｰ—―─━ー]+', 'ー', s)  # normalize choonpus
    s = re.sub('[~∼∾〜〰～]', '', s)  # remove tildes
    s = s.translate(
        maketrans('!"#$%&\'()*+,-./:;<=>?@[¥]^_`{|}~｡､･｢｣',
              '！”＃＄％＆’（）＊＋，－．／：；＜＝＞？＠［￥］＾＿｀｛｜｝〜。、・「」'))

    s = remove_extra_spaces(s)
    s = unicode_normalize('！”＃＄％＆’（）＊＋，－．／：；＜＞？＠［￥］＾＿｀｛｜｝〜', s)  # keep ＝,・,「,」
    s = re.sub('[’]', '\'', s)
    s = re.sub('[”]', '"', s)
    return s

# 同じ名前の層のみパラメータをコピー        
def copy_model(src, dst):
    assert isinstance(src, Chain)
    assert isinstance(dst, Chain)
    for child in src.children():
        if child.name not in dst.__dict__: continue
        dst_child = dst[child.name]
        if type(child) != type(dst_child): continue
        if isinstance(child, Chain):
            copy_model(child, dst_child)
        if isinstance(child, chainer.Link):
            match = True
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                if a[0] != b[0]:
                    match = False
                    break
                if a[1].data.shape != b[1].data.shape:
                    match = False
                    break
            if not match:
                print('Ignore %s because of parameter mismatch' % child.name)
                continue
            for a, b in zip(child.namedparams(), dst_child.namedparams()):
                b[1].data = copy.deepcopy(a[1].data)
            print('Copy %s' % child.name)

def load_w2v(path):
    print('Loading word2vec model...')
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.normpath(os.path.join(base, path))
    model_w2v = word2vec.Word2Vec.load(path)
    print("Done.")
    return model_w2v

def to_variable(texts, model):
    variables = []
    m = MeCab.Tagger("-Owakati")
    for text in texts:
        text = normalize_neologd(text)
        text_sp = m.parse(text)
        # print(text_sp)
        words = text_sp.split()

        text_w2v = []
        for word in words:
            try:
                text_w2v.append(model[word])
            except KeyError:
                pass

        if text_w2v != []:
            variables.append(Variable(np.asarray(text_w2v, dtype=np.float32)))
    
    return variables

def load_models(context):
    base = os.path.dirname(os.path.abspath(__file__))
    base = os.path.normpath(os.path.join(base, '../data/models/'))

    # RNN_Bottom
    models_bottom = []
    for i in range(5):
        model_bottom = models.RNN_SINGLE()
        serializers.load_npz(base+'/bottom/nsteplstm'+str(i)+'best.model', model_bottom)
        models_bottom.append(model_bottom)

    # other
    base = os.path.normpath(os.path.join(base, './context'+str(context)+'/'))

    models_top = []
    models_ova = []
    models_enova = []
    for i in range(5):
        model_top = models.RNN_FINETUNING()
        model_ova = models.RNN_TOP()
        model_enova = models.RNN_CONNECT_AT()

        serializers.load_npz(base+'/top/nsteplstm'+str(i)+'best.model', model_top)
        serializers.load_npz(base+'/ova/nsteplstm'+str(i)+'best.model', model_ova)
        serializers.load_npz(base+'/enova/nsteplstm'+str(i)+'best.model', model_enova)        

        models_top.append(model_top)
        models_ova.append(model_ova)
        models_enova.append(model_enova)

    return models_bottom, models_top, models_ova, models_enova

def print_answers(ys, ys_max):
    print(np.squeeze(ys))
    print([consts.ACTS[m] for m in ys_max])

    prob = [y[m] for y,m in zip(np.squeeze(np.asarray(ys)),ys_max)]
    print(prob)

    answer = ys_max[int(np.argmax(prob))]
    print()

    return int(answer)