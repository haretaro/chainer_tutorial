import numpy as np
from chainer import Variable
import chainer.functions as F

#3入力2つのミニバッチ
x = Variable(np.array([[1,1,1],[2,2,2]],dtype=np.float32))

#出力関数
y = x**2 -2*x + 1

#入力が配列の時は初期誤差を手動で設定する
y.grad = np.ones((2,3), dtype=np.float32)

#誤差逆伝搬で勾配を計算
y.backward()

#勾配を表示.
print('grad',x.grad)
