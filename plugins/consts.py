# 定数
GPU = -1       # 使用:0 不使用:-1
EPOCH = 400
BATCH_SIZE = 400
CONTEXT = 2    # 考慮する文脈の数
LAYERS = 1     # nsteplstmの層の数
IN = 200       # 入力次元数
UNITS = 200    # nsteplstmの出力次元数
DROPOUT = 0.5  # ドロップアウト率
L1_REGULARIZATION = -1 # L1正則化を使うかどうか
L2_REGULARIZATION = -1 # L2正則化を使うかどうか
UNDERSAMPLING = 100000  # ある対話行為の上限サンプル数
CLEAN = 0 # 使用:0 不使用:-1
CHANGE = -1   # 話者交代の情報を使う:0 使わない:-1
CHANGE_MODE = -1 # 話者交代のタイミングで1:-1 話者によって変更:0
MECAB = -1 # MeCab使用:0 Sentence Piece使用:-1
EARLY = 40 # early stopping
TYPE = 1 # 0:9次元のみ 1:9+200次元 2:9+800次元
BEST = 0 # 今までのベストモデル使用:0 不使用:-1

VERSION = '0/' # 下段RNNのうちどれを使うか
FILENAME = '0' # 上段RNNのうちどれを使うか
FILE = '0' # FILENAME内の2値分類器
CONNECT = '1' # 何番目のファイルか

CONST_TEST = 'Test Ok!'

W2V_PATH = r'../data/word2vec/wiki2.model'
ACTS = ['自己開示', '質問(Yes/No)', '質問(What)', '応答(Yes/No)', '応答(平叙)', 'あいづち', 'フィラー', '確認', '要求']