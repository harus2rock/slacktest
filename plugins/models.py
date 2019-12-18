import chainer.functions as F
import chainer.links as L
from chainer import Chain

from plugins import consts

# RNN_Bottom用
class RNN_SINGLE(Chain):
    def __init__(self, n_layers=consts.LAYERS, n_w2vlen=consts.IN, n_units=consts.UNITS, n_tag=9, dropout=consts.DROPOUT):
        """
        n_tag    : ラベルの次元
        """
        super(RNN_SINGLE, self).__init__()
        
        # パラメータを持つ層の登録
        with self.init_scope():
            self.xh = L.NStepBiLSTM(n_layers, n_w2vlen, n_units, dropout)
            self.hh = L.Linear(n_units*2, 100*2)
            self.hy = L.Linear(100*2, n_tag)
        
    def __call__(self, xs):
        """
        x       : list(Variable)
        x.shape : [(文書の単語数, 200)] * batchsize
        y       : Variable
        y.shape : batchsize * 9
        """
        #  FILENAME=0
        hy, _, _ = self.xh(None, None, xs)
        h = F.concat(F.concat(F.split_axis(hy, 2, axis=0),axis=2),axis=0)
        y = F.dropout(F.relu(self.hh(h)))
        y = self.hy(y)
        return h,y

# RNN_Top用
class RNN_FINETUNING(Chain):
    def __init__(self, n_layers=consts.LAYERS, n_in=consts.IN*2, n_units=consts.UNITS*2, n_tag=9, dropout=consts.DROPOUT):
        super(RNN_FINETUNING, self).__init__()
        
        # パラメータを持つ層の登録
        with self.init_scope():
            self.xh = L.NStepBiLSTM(n_layers, n_in, n_units, dropout)
            self.hy1 = L.Linear(n_units*2, 100*2)
            self.hy2 = L.Linear(100*2, n_tag)
            
    def __call__(self, xs):
        """
        xs : list(Variable)
        y  : xp.array
        """
        hy, _, _ = self.xh(None, None, xs)
        h = F.relu(F.concat(F.concat(F.split_axis(hy, 2, axis=0),axis=2),axis=0))
        h = F.dropout(F.relu(self.hy1(h)))
        y = self.hy2(h)
        return y
    
# 2値分類層
class RNN_TOP(Chain):
    def __init__(self, n_layers=consts.LAYERS, n_in=consts.IN*2, n_units=consts.UNITS*2, n_tag=9, dropout=consts.DROPOUT):
        super(RNN_TOP, self).__init__()
        
        # パラメータを持つ層の登録
        with self.init_scope():
            self.xh = L.NStepBiLSTM(n_layers, n_in, n_units, dropout)
            self.h_100_self = L.Linear(n_units*2, 100*2)
            self.h_1_self = L.Linear(100*2,1)
            self.h_100_qyn = L.Linear(n_units*2, 100*2)
            self.h_1_qyn = L.Linear(100*2,1)
            self.h_100_qw = L.Linear(n_units*2, 100*2)
            self.h_1_qw = L.Linear(100*2,1)
            self.h_100_ayn = L.Linear(n_units*2, 100*2)
            self.h_1_ayn = L.Linear(100*2,1)
            self.h_100_aw = L.Linear(n_units*2, 100*2)
            self.h_1_aw = L.Linear(100*2,1)
            self.h_100_res = L.Linear(n_units*2, 100*2)
            self.h_1_res = L.Linear(100*2,1)
            self.h_100_fil = L.Linear(n_units*2, 100*2)
            self.h_1_fil = L.Linear(100*2,1)
            self.h_100_con = L.Linear(n_units*2, 100*2)
            self.h_1_con = L.Linear(100*2,1)
            self.h_100_req = L.Linear(n_units*2, 100*2)
            self.h_1_req = L.Linear(100*2,1)
            
    def __call__(self, xs):
        """
        xs : list(Variable)
        y  : xp.array
        """
        hy, _, _ = self.xh(None, None, xs)
        h = F.relu(F.concat(F.concat(F.split_axis(hy, 2, axis=0),axis=2),axis=0))

        h_self = F.dropout(F.relu(self.h_100_self(h)))
        y_self = self.h_1_self(h_self)

        h_qyn = F.dropout(F.relu(self.h_100_qyn(h)))
        y_qyn = self.h_1_qyn(h_qyn)

        h_qw = F.dropout(F.relu(self.h_100_qw(h)))
        y_qw = self.h_1_qw(h_qw)

        h_ayn = F.dropout(F.relu(self.h_100_ayn(h)))
        y_ayn = self.h_1_ayn(h_ayn)

        h_aw = F.dropout(F.relu(self.h_100_aw(h)))
        y_aw = self.h_1_aw(h_aw)

        h_res = F.dropout(F.relu(self.h_100_res(h)))
        y_res = self.h_1_res(h_res)

        h_fil = F.dropout(F.relu(self.h_100_fil(h)))
        y_fil = self.h_1_fil(h_fil)

        h_con = F.dropout(F.relu(self.h_100_con(h)))
        y_con = self.h_1_con(h_con)

        h_req = F.dropout(F.relu(self.h_100_req(h)))
        y_req = self.h_1_req(h_req)

        y = F.concat((y_self, y_qyn, y_qw, y_ayn, y_aw, y_res, y_fil, y_con, y_req))

        return y

class RNN_CONNECT_AT(Chain):
    def __init__(self, n_layers=consts.LAYERS, n_in=consts.IN*2, n_units=consts.UNITS*2, dropout=consts.DROPOUT):
        super(RNN_CONNECT_AT,self).__init__()
        
        with self.init_scope():
            self.xh = L.NStepBiLSTM(n_layers, n_in, n_units, dropout)
            self.hy1 = L.Linear(n_units*2, 100*2)
            
            self.h_100_self = L.Linear(n_units*2, 100*2)
            self.h_100_qyn = L.Linear(n_units*2, 100*2)
            self.h_100_qw = L.Linear(n_units*2, 100*2)
            self.h_100_ayn = L.Linear(n_units*2, 100*2)
            self.h_100_aw = L.Linear(n_units*2, 100*2)
            self.h_100_res = L.Linear(n_units*2, 100*2)
            self.h_100_fil = L.Linear(n_units*2, 100*2)
            self.h_100_con = L.Linear(n_units*2, 100*2)
            self.h_100_req = L.Linear(n_units*2, 100*2)
            
            self.at = L.Linear(100*2, 1)
#            self.at1 = L.Linear(100*2, 100) # 200->100->1じゃなくて200->1でもいいかも
#            self.at2 = L.Linear(100, 1)
            
            self.out = L.Linear(100*2, 9) # 200->9じゃなくて200->100->9でもいいかも
#            self.out1 = L.Linear(100*2,100)
#            self.out2 = L.Linear(100, 9)
            
    def __call__(self, xs):
        """
        xs : list(Variable)
        y  : Variable
        """
        # _/_/_/RNN_Top(Bi-LSTM)
        hy, _, _ = self.xh(None, None, xs) # shape:(2,batch,units)
        h1 = F.relu(F.concat(F.concat(F.split_axis(hy, 2, axis=0),axis=2),axis=0)) # shape:(batch,units*2)

        # relu
        # _/_/_/FC
        h = F.relu(self.hy1(h1)) #shape:(batch,100*2)
        h = F.split_axis(h, h.shape[0],axis=0) #tuple:(batch,1,100*2)
        
        # _/_/_/FC for twoclass
#        h_self = F.relu(self.h_100_self(h1)) #shape:(batch,100*2)
#        h_self = F.split_axis(h_self, h_self.shape[0],axis=0) #tuple:(batch,1,100*2)
#        h_qyn = F.relu(self.h_100_qyn(h1))
#        h_qyn = F.split_axis(h_qyn, h_qyn.shape[0],axis=0)
#        h_qw = F.relu(self.h_100_qw(h1))
#        h_qw = F.split_axis(h_qw, h_qw.shape[0],axis=0)
#        h_ayn = F.relu(self.h_100_ayn(h1))
#        h_ayn = F.split_axis(h_ayn, h_ayn.shape[0],axis=0)
#        h_aw = F.relu(self.h_100_aw(h1))
#        h_aw = F.split_axis(h_aw, h_aw.shape[0],axis=0)
#        h_res = F.relu(self.h_100_res(h1))
#        h_res = F.split_axis(h_res, h_res.shape[0],axis=0)
#        h_fil = F.relu(self.h_100_fil(h1))
#        h_fil = F.split_axis(h_fil, h_fil.shape[0],axis=0)
#        h_con = F.relu(self.h_100_con(h1))
#        h_con = F.split_axis(h_con, h_con.shape[0],axis=0)
#        h_req = F.relu(self.h_100_req(h1))
#        h_req = F.split_axis(h_req, h_req.shape[0],axis=0)
        
        # reluなし
        # _/_/_/FC
#        h = self.hy1(h1) #shape:(batch,100*2)
#        h = F.split_axis(h, h.shape[0],axis=0) #tuple:(batch,1,100*2)
        
        # _/_/_/FC for twoclass
        h_self = self.h_100_self(h1) #shape:(batch,100*2)
        h_self = F.split_axis(h_self, h_self.shape[0],axis=0) #tuple:(batch,1,100*2)
        h_qyn = self.h_100_qyn(h1)
        h_qyn = F.split_axis(h_qyn, h_qyn.shape[0],axis=0)
        h_qw = self.h_100_qw(h1)
        h_qw = F.split_axis(h_qw, h_qw.shape[0],axis=0)
        h_ayn = self.h_100_ayn(h1)
        h_ayn = F.split_axis(h_ayn, h_ayn.shape[0],axis=0)
        h_aw = self.h_100_aw(h1)
        h_aw = F.split_axis(h_aw, h_aw.shape[0],axis=0)
        h_res = self.h_100_res(h1)
        h_res = F.split_axis(h_res, h_res.shape[0],axis=0)
        h_fil = self.h_100_fil(h1)
        h_fil = F.split_axis(h_fil, h_fil.shape[0],axis=0)
        h_con = self.h_100_con(h1)
        h_con = F.split_axis(h_con, h_con.shape[0],axis=0)
        h_req = self.h_100_req(h1)
        h_req = F.split_axis(h_req, h_req.shape[0],axis=0)
        
        # _/_/_/全結合層1層目を全部くっつける
        hs = [] # shape:(b,10,100*2)
        for h1,s,qy,qw,ay,aw,r,f,c,r in zip(h,h_self,h_qyn,h_qw,h_ayn,h_aw,h_res,h_fil,h_con,h_req):
            hs.append(F.concat([h1,s,qy,qw,ay,aw,r,f,c,r], axis=0))
        
        """
        A : 内積取らずにattention計算
        """
        
        # _/_/_/attention計算
        concat_hs = F.concat(hs, axis=0) # (10*b,100*2)
        # 1層でattention計算
        attn = F.tanh(self.at(concat_hs))
        # 2層でattention計算
#        attn = F.relu(self.at1(concat_hs)) # leaky_reluでもいいかも
#        attn = self.at2(attn) # (10*b,1)
        
        sp_attn = F.split_axis(attn, len(hs), axis=0) # tuple:(b,10,1)
        sp_attn_pad = F.pad_sequence(sp_attn, padding=-1024.0) #(b,10,1)
        attn_softmax = F.softmax(sp_attn_pad, axis=1)
                
        # _/_/_/形をそろえてbroadcast
        hs_pad = F.pad_sequence(hs, length=None, padding=0.0) # (b,10,100*2)
        hs_pad_reshape = F.reshape(hs_pad, (-1, hs_pad.shape[-1])) # (10*b,100*2)
        
        attn_softmax_reshape = F.broadcast_to(F.reshape(attn_softmax, (-1, attn_softmax.shape[-1])), hs_pad_reshape.shape) # (10*b,100*2)
        
        # _/_/_/attentionを出力にかけてsoftmax方向に足す！
        attention_hidden = hs_pad_reshape * attn_softmax_reshape # (10*b,100*2)
        attention_hidden_reshape = F.reshape(attention_hidden, (len(hs), -1, attention_hidden.shape[-1])) # (b,10,100*2)
        
        result = F.sum(attention_hidden_reshape, axis=1) # (b,100*2)
        
        """
        B : 内積とってattention計算
        """
#        hs = F.concat([F.expand_dims(h, axis=0) for h in hs], axis=0) # (b,10,100*2)
#        # _/_/_/attention計算
#        score = F.batch_matmul(hs, hs, transb=True)
#        scale_score = 1. / UNITS ** 0.5
#        score = score * scale_score # scaled dot-product
#        attention = F.softmax(score, axis=2)
#        
#        # _/_/_/加重平均とってsoftmax方向に足してtanh！
#        c = F.batch_matmul(attention, hs)
#        result = F.tanh(F.sum(c, axis=1))
        
        """
        出力
        """
        
        # _/_/_/出力層
        # 1層で出力
        y = self.out(result)
        # 2層で出力
#        y = F.dropout(F.relu(self.out1(result)))
#        y = self.out2(y)
        return y

import datetime

class output_text :
    # 初期処理
    def __init__(self) :
        print("start")

    # 日付・時刻を出力する
    def output_date(self, mode = 0) :
        if mode == 1 :
            # 
            print(datetime.date.today())
        elif mode == 2 :
            print(datetime.datetime.today())
        else :
            print("none")