import numpy as np
import torch

def openreadtxt(file_name):
    x = []
    y = []
    file = open(file_name,'r')  #打开文件
    file_data = file.readlines() #读取所有行
    # [[4, 4, 8, 2, 2, 2, 4, 2, 4, 2, 6, 4, 2, 4, 4, 6, 8, 6, 2, 2, 6, 8, 8, 2, 6, 6, 2, 6], 0.6480000019073486]

    cnt = 0

    for row in file_data:
        row = row.replace(']', '')
        tmp_list = row.replace('[', '')
        tmp_list = row.split(' ') #按‘，’切分每行的数据
        
        tmp_list[-1] = tmp_list[-1].replace('\n',',') #去掉换行符
        tmp_list[0] = tmp_list[0].replace('[', '')
        for i in range(len(tmp_list)-1):
            tmp_list[i] = float(tmp_list[i].replace(',', ''))
        
        tmp_list[-1] = float(tmp_list[-1].replace(',', ''))
        x.append(tmp_list[:-1]) #将每行数据插入data中
        y.append(tmp_list[-1])


        cnt = cnt + 1
        # if cnt >=2:
        #     break
    print('数据集总共载入',cnt,'个样本！')
    return np.array(x, dtype='float32'), np.array(y, dtype='float32')
 
 
if __name__=="__main__":
    x, y = openreadtxt('acc_all')
    print('x.type:', type(x))
    print('x.shape:',x.shape)
    print('y:',y)