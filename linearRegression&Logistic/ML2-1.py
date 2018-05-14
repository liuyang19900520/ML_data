import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq  ##引入最小二乘法算法

'''
    创建体重y，身高x
'''
Yi = np.array([61, 57, 58, 40, 90, 35, 45])
Xi = np.array([170, 168, 175, 153, 185, 135, 172])


'''
    设定拟合函数和偏差函数
    函数的形状确定过程：
    1.先画样本图像
    2.根据样本图像大致形状确定函数形式(直线、抛物线、正弦余弦等)
'''

##需要拟合的函数func :指定函数的形状
def func(p,x):
    k,b=p
    return k*x+b

##偏差函数：x,y都是列表:这里的x,y更上面的Xi,Yi中是一一对应的
# 我们的目标就是不断调整k和b使得error不断减小。
# 这里的error函数和神经网络中常说的cost函数实际上是一回事，只不过这里更简单些而已。
def error(p,x,y):
    return func(p,x)-y

'''
    主要部分：附带部分说明
    1.leastsq函数的返回值tuple，第一个元素是求解结果，第二个是求解的代价值(个人理解)
    2.官网的原话（第二个值）：Value of the cost function at the solution
    3.实例：Para=>(array([ 0.61349535,  1.79409255]), 3)
    4.返回值元组中第一个值的数量跟需要求解的参数的数量一致
'''

#k,b的初始值，可以任意设定,经过几次试验，发现p0的值会影响cost的值：Para[1]
#p0里放的是k、b的初始值，这个值可以随意指定。
# 往后随着迭代次数增加，k、b将会不断变化，使得error函数的值越来越小。
p0=[4,20]

#把error函数中除了p0以外的参数打包到args中(使用要求)
Para=leastsq(error,p0,args=(Xi,Yi))

#读取结果
k,b=Para[0]
print("k=",k,"b=",b)
print("cost："+str(Para[1]))
print("求解的拟合直线为:")
print("y="+str(round(k,2))+"x+"+str(round(b,2)))

#p0=[0,0]
#k= 0.9076449857973351 b= -95.00755623113787
#cost：3
#y=0.91x+-95.01

#p0=[4,20]
#k= 0.9076449912059097 b= -95.0075571180014
#cost：1
#y=0.91x+-95.01


'''
   绘图，看拟合效果.
   matplotlib默认不支持中文，label设置中文的话需要另行设置
   如果报错，改成英文就可以
'''

#画样本点
plt.figure(figsize=(8,6)) ##指定图像比例： 8：6
plt.scatter(Xi,Yi,color="green",label="样本数据",linewidth=2)

#画拟合直线
x=np.linspace(0,200,100) ##在0-15直接画100个连续点
y=k*x+b ##函数式
plt.plot(x,y,color="red",label="拟合直线",linewidth=2)
plt.legend(loc='lower right') #绘制图例
plt.show()


# R语言的summary
# df = pd.DataFrame(y, x)
# print("============================")
# # print(df.corr())
# # print( df.info())
# print(df.describe())
# print("============================")
# print("Lower noise", pearsonr(x, y))
# print("Higher noise", pearsonr(x, y))
