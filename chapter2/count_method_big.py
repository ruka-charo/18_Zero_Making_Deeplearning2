import os, sys
os.chdir('/Users/rukaoide/Library/Mobile Documents/com~apple~CloudDocs/Documents/Python/18_Zero_Making_Deeplearning2/chapter2')
sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb
from sklearn.utils.extmath import randomized_svd


window_size = 2
wordvec_size = 100

# テキストを共起行列にし、PPMI行列に変換する
corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting  co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

# SVDを実行
print('calculating SVD ...')
try:
    # truncated SVD (fast!)
    U, S, V = randomized_svd(W, n_components=wordvec_size, n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(W)


# 次元削減後の単語ベクトル
word_vecs = U[:, :wordvec_size]

# 関連性の高い単語TOP5を表示
querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)
    
