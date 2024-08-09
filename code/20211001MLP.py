import pandas as pd
import numpy as np
import time
from sklearn import ensemble as eb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics as mt
from matplotlib import pyplot as plt


def plot_train_test(y_train,y_test,predict_train,predict_test):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(y_train.shape[0])),predict_train,'c*-',label='predicted Value')
    ax.plot(list(range(y_train.shape[0])),y_train,'m.-',label='True Value')
    plt.title('Comparison of training set prediction results')
    plt.ylabel('Sample Value')
    plt.xlabel('Sample Number')
    plt.legend()
    plt.show()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(list(range(y_test.shape[0])),predict_test,'c*-',label='predicted Value')
    ax.plot(list(range(y_test.shape[0])),y_test,'m.-',label='True Value')
    plt.title('Comparison of testing set prediction results')
    plt.ylabel('Sample Value')
    plt.xlabel('Sample Number')
    plt.legend()
    plt.show()



fi = "test1.xlsx"
ds = pd.read_excel(fi, 'sheet1',header = None) 
stas = np.zeros((192,8),dtype=float)
#formula = np.zeros((384,10),dtype='U8')
TP = np.zeros((52,768),dtype=float)
for n in range(0,192):
   data=ds[1:]
   datset= np.array(data)
   X=datset[:,0:16]
   Y=datset[:,n+16,np.newaxis]
   #datset_select=np.concatenate((X,Y),axis=1) 
   
   X_train, X_test, y_train, y_test=train_test_split(X,Y,test_size=0.2,random_state=12343)
    #对数据归一化
   sc_train_X = MinMaxScaler(feature_range=(0, 1)) 
   sc_train_y = MinMaxScaler(feature_range=(0, 1))   
   X_train_Norm = sc_train_X.fit_transform(X_train)
   X_test_Norm = sc_train_X.transform(X_test)
   y_train_Norm = sc_train_y.fit_transform(y_train)
   y_test_Norm = sc_train_y.transform(y_test) 

    #fit函数需传入y值为一维数组
   y_train_Norm_array = np.array(y_train_Norm).flatten()
   y_test_Norm_array = np.array(y_test_Norm).flatten()
    
     #建立MLP模型
    model_mlp = nn.MLPRegressor(\
        hidden_layer_sizes=(35,100,100),activation='relu',solver='adam', alpha=0.0001,batch_size='auto',
        learning_rate='adaptive',learning_rate_init=0.001, max_iter=5000, shuffle=True,
        random_state=1, tol=0.0001, verbose=False, warm_start=False,
        early_stopping=False,beta_1=0.9, beta_2=0.999, epsilon=1e-08)

   #开始训练并计
   startTime = time.time() 
   model_mlp.fit(X_train_Norm, y_train_Norm_array)
   
    #调用sklearn库函数，并计算模型mse值并输出
   mlp_score=model_mlp.score(X_test_Norm,y_test_Norm_array)
   #print('sklearn梯度提升决策树-回归模型mse值',gbdt_score)
    
    #输出模型训练用时
   stopTime = time.time()
   sumTime = stopTime - startTime
   print('总时间是：', sumTime)
    
    #反归一化保存预测值与真实值
   predict_train_Norm = model_mlp.predict(X_train_Norm)
   predict_train = sc_train_y.inverse_transform(np.matrix(predict_train_Norm.tolist()).T)
   predict_test_Norm = model_mlp.predict(X_test_Norm)
   predict_test = sc_train_y.inverse_transform(np.matrix(predict_test_Norm.tolist()).T)
   
  
   #计算MSE值
   mse_train = mt.mean_squared_error(y_train,predict_train)
   mse_test = mt.mean_squared_error(y_test,predict_test)
   #print('训练集真实值与预测值MSE值',mse_train)
   #print('测试集真实值与预测值MSE值',mse_test)
   


   
    #计算RMSE值
   rmse_train = mt.mean_squared_error(y_train,predict_train,squared = False)
   rmse_test = mt.mean_squared_error(y_test,predict_test,squared = False)
   #print('训练集真实值与预测值RMSE值',rmse_train)
   #print('测试集真实值与预测值RMSE值',rmse_test)
    
    #计算MAE值
   mae_train = mt.mean_absolute_error(y_train,predict_train)
   mae_test = mt.mean_absolute_error(y_test,predict_test)
   #print('训练集真实值与预测值MAE值',mae_train)
   #print('测试集真实值与预测值MAE值',mae_test)
    
    #计算R^2值
   r2_train = mt.r2_score(y_train,predict_train)
   r2_test = mt.r2_score(y_test,predict_test)
   #print('训练集真实值与预测值R^2值',r2_train)
   #print('测试集真实值与预测值R^2值',r2_test)
   
   
   stas[n,0]= r2_train
   stas[n,1]= r2_test
   stas[n,2]= mse_train
   stas[n,3]= mse_test
   stas[n,4]= rmse_train
   stas[n,5]= rmse_test
   stas[n,6]= mae_train
   stas[n,7]= mae_test
   print(stas)
    
    #将预测值和真实值放在同一个矩阵变量中对比
   #result_train = np.hstack((y_train,predict_train))
   #result_test = np.hstack((y_test,predict_test))
  
    # #将训练集真实值和预测值数据并写入文件
   #data_train = pd.DataFrame(result_train)
 
 
   #data_train.columns = ['真实值','预测值']
   #writer_train = pd.ExcelWriter('GBDT_data_train.xlsx')
   #data_train.to_excel(writer_train)  
   #writer_train.save()  
    
    # #将测试集真实值和预测值数据并写入文件
   #data_test = pd.DataFrame(result_test)
   #data_test.columns = ['真实值','预测值']
   #writer_test = pd.ExcelWriter('GBDT_data_test.xlsx')
   #data_test.to_excel(writer_test)  
   #writer_test.save()
   
   lens1=y_train.shape[0]
   lens2=y_test.shape[0]
   T1=np.squeeze(y_train[:]) 
   P1=np.squeeze(predict_train[:])
   T2=np.squeeze(y_test[:])
   P2=np.squeeze(predict_test[:])
   TP[0:lens1,4*n]=T1
   TP[0:lens1,4*n+1]=P1
   TP[0:lens2,4*n+2]=T2
   TP[0:lens2,4*n+3]=P2
   print(TP)
    #绘制图像
   plot_train_test(y_train,y_test,predict_train,predict_test)

import xlwt
wbk=xlwt.Workbook()
sheet = wbk.add_sheet(u'sheet1',cell_overwrite_ok=True)
A1=stas
data = pd.DataFrame(A1)
writer = pd.ExcelWriter('A1.xlsx')		# 写入Excel文件
data.to_excel(writer, '评价指标', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.close()

import xlwt
wbk=xlwt.Workbook()
sheet = wbk.add_sheet(u'sheet1',cell_overwrite_ok=True)
#print(TP)
#A2 = np.array(TP)
A2=TP
#print(A2[0,:])
data = pd.DataFrame(A2)
writer = pd.ExcelWriter('A2.xlsx')		# 写入Excel文件
data.to_excel(writer, '预测值', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
writer.close()

