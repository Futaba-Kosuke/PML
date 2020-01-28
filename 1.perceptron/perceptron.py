import numpy as np

class Perceptron(object):

    def __init__ (self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit (self, X, y):
        rgen = np.random.RandomState(self.random_state)
        # 標準偏差0.01の正規分布
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                # target: -1, self.predict(x): 1 => update = -2 * self.eta
                # target: -1, self.predict(x): -1 => update = 0
                # target: 1, self.predict(x): 1 => update = 0
                # target: 1, self.predict(x): -1 => update = 2 * self.eta
                # => targetを-1, 1で初期化して入力しなければならない
                update = self.eta * (target - self.predict(xi))
                # 重みの更新
                self.w_[1:] += update * xi
                # バイアスの更新
                self.w_[0] += update
                # 誤差の計算
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self
    
    def net_input (self, X):
        # Affineする
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict (self, X):
        # 正負の符号行列を生成
        return np.where(self.net_input(X) >= 0.0, 1, -1)