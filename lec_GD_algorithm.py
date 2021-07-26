import numpy as np
import matplotlib.pyplot as plt

def make_linear(w=0.5, b=0.8, size=50, noise=1.0):
    x=np.random.rand(size)
    y = w * x + b #0.5x+0.8
    noise = np.random.uniform(-abs(noise), abs(noise), size=y.shape)
    yy = y+ noise #노이즈 들어간 y
    plt.figure(figsize=(10,7))
    plt.plot(x, y, color="r", label=f"y={w}*x+{b}")
    plt.scatter(x, yy, label="data")
    plt.legend(fontsize=20)
    plt.show()
    print(f"y={w}*x+{b}")
    return x, yy

x, y =make_linear(w=0.3, b=0.5, size=50, noise=0.01)

# 초기값(initializer)과 y_hat(예측, prediction)함수 정의
w=np.random.uniform(low=-1.0, high=1.0)
b=np.random.uniform(low=-1.0, high=1.0)
print(w, b)

# 가설 함수
y_hat = w * x + b # 예측값 100개 -0.75x - 0.14

# 비용 함수(cost func.) 또는 손실 함수(loss func.)
# Mean Squared Error:잔차 제곱 합의 평균
error = ((y_hat-y)**2).mean()

#Gradient Descent 구현 (단항식)
num_epoch = 300 #반복 횟수
learning_rate = 0.01

w=np.random.uniform(low=-1.0, high=1.0)
b=np.random.uniform(low=-1.0, high=1.0)

errors=[]
for epoch in range(num_epoch):
    y_hat = w * x + b #편미분한 값 재적용
    if error < 0.0006: #EarlyStop:조기종료
        break

    #편미분
    w = w - learning_rate * ((y_hat-y)*x).mean()
    b = b - learning_rate * (y_hat-y).mean

    errors.append(error)

    if epoch % 5 == 0:
        print("{0.2}w={1:.5f}, b = {2:.5f} error = {3:.5f}".format(epoch, w, b, error))

print("----"*15)
print("{0.2}w={1:.5f}, b = {2:.5f} error = {3:.5f}".format(epoch, w, b, error))
