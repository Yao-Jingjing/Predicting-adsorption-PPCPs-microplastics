from __future__ import division
import numpy as np
from minepy import MINE
import pandas as pd
import matplotlib.pyplot as plt

fi = "test1.xlsx"
ds = pd.read_excel(fi, 'sheet1',header = None)
#stas = np.zeros((2,3),dtype=float)

for n in range(0,2): 
    for m in range(0,1): 
        data=ds[1:]
        datset= np.array(data)
        x=datset[:,n]
        y=datset[:,m+16]
        mine = MINE(alpha=0.6, c=15)
        mine.compute_score(x, y)
        print(n+1,m+16,"MIC", mine.mic())
   
#import xlwt
#wbk=xlwt.Workbook()
#sheet = wbk.add_sheet(u'sheet1',cell_overwrite_ok=True)
#A1=stas
#data = pd.DataFrame(A1)
#writer = pd.ExcelWriter('A1.xlsx')		# 写入Excel文件
#data.to_excel(writer, '评价指标', float_format='%.5f')		# ‘page_1’是写入excel的sheet名
#writer.close()