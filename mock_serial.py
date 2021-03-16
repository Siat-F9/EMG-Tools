import pyqtgraph as pg
import array
import threading
import numpy as np
from queue import Queue
from scipy import signal
import time
from scipy.fftpack import fft
import scipy.io as sio
import serial





i = 0
q = Queue(maxsize=0)
count =0
count_plot = 0

subject = 2
gesture = 1
trial = 1

mean_threshold = 0
mean_value = 0

SAVE_LEN = 5000
RECORD_TIME = SAVE_LEN // 1000
is_record = False
rec_count = 0
remain_time = RECORD_TIME
MAX_COUNT = 50*RECORD_TIME #每秒是50个数据

#set threshold in order to ensure muscle contraction
is_set_thres = False
rec_count_thres = 0
remain_time_thres = RECORD_TIME
MAX_COUNT = 50*RECORD_TIME #每秒是50个数据

wins_snapshot = []

global F1;
global M;
global lamda;
global I;
global c;
global x_noise;
global P_last;
global w_last;
global is_draw;
global label;

F1 = 50
F2 = 100
F3 = 150
F4 = 200
F5 = 250
F6 = 300
F7 = 350
F8 = 400
F9 = 450
F10 = 500
M = 4 #%定义FIR滤波器阶数
lamda = 0.9 #定义遗忘因子
I = np.eye(M) #生成对应的单位矩阵
c = 0.01 #小正数 保证矩阵P非奇异
t = np.arange(1000)/1000
t = t.reshape(1000,1)

x_noise = np.hstack((np.sin(2*np.pi*F1*t),np.cos(2*np.pi*F1*t),np.sin(2*np.pi*F2*t),np.cos(2*np.pi*F2*t),
#np.sin(2*np.pi*F3*t),np.cos(2*np.pi*F3*t),np.sin(2*np.pi*F4*t),np.cos(2*np.pi*F4*t),
#np.sin(2*np.pi*F5*t),np.cos(2*np.pi*F5*t),np.sin(2*np.pi*F6*t),np.cos(2*np.pi*F6*t),
#np.sin(2*np.pi*F7*t),np.cos(2*np.pi*F7*t),np.sin(2*np.pi*F8*t),np.cos(2*np.pi*F8*t),
#np.sin(2*np.pi*F9*t),np.cos(2*np.pi*F9*t),np.sin(2*np.pi*F10*t),np.cos(2*np.pi*F10*t)
));
P_last = I/c;
w_last = np.zeros([M,1]);

b, a = signal.butter(8, 2*10/1000, 'highpass') #filter direct ciruit

data3 = array.array('i')  # 可动态改变数组的大小,double型数组
data3=np.zeros((8,SAVE_LEN)).__array__('d')#把数组长度定下来

def rls_algo(sample):
    global F1;
    global M;
    global lamda;
    global I;
    global c;
    global x_noise;
    global P_last;
    global w_last;
    sample1 = np.zeros(len(sample))
    for i in range(len(sample)):
        d = sample[i];                         #输入新的期望信号
        x = x_noise[i,:].reshape(M,1);                 #输入新的信号矢量
        tmp = np.dot(x.T,P_last)
        tmp = np.dot(tmp,x)
        
        #if i == 0: ###### do not remove
        #    print(lamda,tmp,lamda+tmp)

        K = (np.dot(P_last , x))/(lamda + tmp);   #计算增益矢量
        #K = (np.dot(P_last , x))/(lamda + x.T *  P_last * x);   #计算增益矢量
        y = np.dot(-x.T , w_last);                          #计算FIR滤波器输出

        Eta = (d + y);                             #计算估计的误差
        w = w_last + K * Eta;                    #计算滤波器系数矢量
        P = (I - K * x.T)* P_last/lamda;          #计算误差相关矩阵
        P_last = P;
        w_last = w;
        sample1[i] = Eta;
    return sample1

def Serial():
    global i;
    global q;
    head = [-1,-1,-1,-1,-1]
    global count;
    data2 = array.array('i')  # 可动态改变数组的大小,double型数组
    data2=np.zeros((channelNum,historyLength)).__array__('d')#把数组长度定下来


    data_rls2 = array.array('i')  # 可动态改变数组的大小,double型数组
    data_rls2 =np.zeros((channelNum,historyLength)).__array__('d')#把数组长度定下来


    while(True):
        '''
        n = mSerial.inWaiting()
        if(n):
            dat = int.from_bytes(mSerial.readline(1),byteorder='little')  # 格式转换
            if(dat>>7):
                dat =256-dat
                dat =0-dat
            q.put(dat)
        '''
        cur_time = 0.0
        last_time = float('inf')
        global data3
        global is_record
        global rec_count
        global remain_time

        global is_set_thres
        global rec_count_thres
        global remain_time_thres

        global mean_threshold
        global mean_value

        for i in range(5):
            head[i] = mSerial.read(1)
        while True:
            if head[0] == b'\x01' and head[1] == b'\x02' and head[2] == b'\x03' and head[3] == b'\x04' and head[4] == b'\x05':
                count = count + 1
                print('count',count)
                cur_time = time.time()
                if cur_time - last_time > 1: # for record data to form emg dataset
                    print('1111111111')
                    #s = 'Dataset/' + str(subject) + '_' + str(gesture) + '_' + str(trial)
                    #np.save(s,data)
                    #sio.savemat('Dataset/{:03d}_{:03d}_{:03d}.mat'.format(subject,gesture,trial),{'data':data3})
                    #trial = trial + 1
                last_time = cur_time

                message=mSerial.read(320)
                # print(message[0:40])
                emg_serial_data = np.array([ x for x in message ]).reshape(8,40) / 4096 * 3.3
                emg_frame_data = emg_serial_data[:,0::2]*256 + emg_serial_data[:,1::2]
                mean_value = np.mean(abs(emg_frame_data-np.mean(emg_frame_data)))

                data[:,0:980] = data2[:,20:1000]
                data[:,980:1000] = emg_frame_data
                data2 = data

                data3[:,0:SAVE_LEN - 20] = data3[:,20:SAVE_LEN]
                data3[:,SAVE_LEN - 20:SAVE_LEN] = emg_frame_data # 10s data

                #for i in range(channelNum):
                data_rls[0,0:980] = data_rls2[0,20:1000]
                data_rls[0,980:1000] = rls_algo(data[0,980:1000])
                data_rls2 = data_rls

                if is_record == True:
                    print(rec_count)
                    if rec_count != MAX_COUNT: 
                        rec_count = rec_count + 1
                    if rec_count % 50 == 0:
                        remain_time = RECORD_TIME - rec_count // 50

                if is_set_thres == True:
                    print(rec_count_thres)
                    if rec_count_thres != MAX_COUNT: 
                        rec_count_thres = rec_count_thres + 1
                    if rec_count_thres % 50 == 0:
                        remain_time_thres = RECORD_TIME - rec_count_thres // 50
                '''
                if count % 50 ==0:
                    s = 'data/save_' + str(count//50)
                    np.save(s,data)
                    s = 'data/save_rls_' + str(count//50)
                    np.save(s,data_rls[0])
                '''
                for i in range(5):
                    head[i] = mSerial.read(1)
            else:
                head[:-1] = head[1:]
                head[4] = mSerial.read(1)

def plotData():
    '''
    global i;
    if i < historyLength:
        data[i] = q.get()
        i = i+1
    else:
        data[:-1] = data[1:]
        data[i-1] = q.get()
    '''
    global trial
    global gesture
    global subject

    global count
    global count_plot;
    global is_record
    global remain_time
    global rec_count
    global is_set_thres
    global remain_time_thres
    global rec_count_thres
    global mean_value
    global mean_threshold
    if is_record == True:
        if remain_time == 0:
            is_record = False
            remain_time = 5
            rec_count = 0
            #record signal when time lasts for 5s
            sio.savemat('Dataset1/{:03d}_{:03d}_{:03d}.mat'.format(subject,gesture,trial),{'data':data3})
            win = pg.GraphicsWindow()  # 建立窗口
            win.setWindowTitle('肌电8通道波形图 sub:{},ges{},tri{}'.format(subject,gesture,trial))
            win.resize(1000, 800)  # 小窗口大小
            trial = trial + 1
            for i in range(8):
                p = win.addPlot()  # 把图p加入到窗口中
                p.showGrid(x=True, y=True)  # 把X和Y的表格打开
                p.setRange(xRange=[0, SAVE_LEN], yRange=[0, 3.3], padding=0)
                p.setLabel(axis='left', text='y / V')  # 靠左
                curve = p.plot()  # 绘制一个图形
                curve.setData(data3[i])
                win.nextRow()
            wins_snapshot.append(win)

        label.setText(remain_time)

    if is_set_thres == True:
        if remain_time_thres == 0:
            is_set_thres = False
            remain_time_thres = 5
            rec_count_thres = 0
            mean_threshold = np.mean(abs(data3-np.mean(data3)))
            label_threshold.setText(mean_threshold)
        else:
            label_threshold.setText(remain_time_thres)
    
    if mean_threshold != 0 and mean_value > mean_threshold:
        label_mean.setText(mean_value,color='FF0000')
    else:
        label_mean.setText(mean_value,color='0000FF')

    count_plot = count_plot + 1
    print('plot',count_plot)
    for channel in range(len(curves)): 
        curves[channel].setData(data[channel])
    curve4.setData(data_rls[0])
    #print('after filter',data_rls[0,980:1000])
    if count % 50 ==0:
        print('-----> plot freq ')
        Y = fft(data[0]) #ignore 1924
        Y = np.abs(Y)
        curve2.setData(20*np.log10(Y)) 
        Y = fft(data_rls[0])
        Y = np.abs(Y)
        curve3.setData(20*np.log10(Y)) 

def mouseClicked(evt):  # 录入数据
    global is_record

    #mousePoint = p.vb.mapSceneToView(evt[0])
    print('receive click -------------------------->',is_record)
    is_record = True


def mouseClicked2(evt):  # 设置阈值
    global is_set_thres
    print('receive click 222222 -------------------------->',is_set_thres)
    is_set_thres = True

def mouseClicked3(evt):  # 改变手势
    global trial
    global gesture
    global subject
    print('receive click 333333 -------------------------->')
    gesture = gesture + 1
    trial = 1
    win3.setWindowTitle(u'channel 1 滤波后sub:{},ges{},tri{}'.format(subject,gesture,trial))

if __name__ == "__main__":

    app = pg.mkQApp()  # 建立app
    win = pg.GraphicsWindow()  # 建立窗口
    win.setWindowTitle(u'肌电8通道波形图')
    win.resize(900, 800)  # 小窗口大小


    data = array.array('i')  # 可动态改变数组的大小,double型数组
    historyLength = 1000  # 横坐标长度
    channelNum = 8
    a = 0
    data=np.zeros((channelNum,historyLength)).__array__('d')#把数组长度定下来

    data_rls = array.array('i')  # 可动态改变数组的大小,double型数组
    data_rls =np.zeros((channelNum,historyLength)).__array__('d')#把数组长度定下来
    
    curves = []

    label = win.addLabel(RECORD_TIME)
    win.nextRow()
    label_threshold = win.addLabel(0)
    win.nextRow()
    label_mean = win.addLabel(0)
    win.nextRow()

    for channel in range(channelNum):
        p = win.addPlot()  # 把图p加入到窗口中
        p.showGrid(x=True, y=True)  # 把X和Y的表格打开
        p.setRange(xRange=[0, historyLength], yRange=[0, 3.3], padding=0)
        #p.setLabel(axis='left', text='y / V')  # 靠左
        #p.setLabel(axis='bottom', text='x / point')
        #p.setTitle('channel:' + str(channel+1))  # 表格的名字
        curve = p.plot()  # 绘制一个图形
        curve.setData(data[i])
        curves.append(curve)
        win.nextRow()

    # portx = 'COM4'
    portx = '/dev/tty.usbmodemADPT0157431'
    # portx = '/dev/tty.usbserial-AR0K409Y'
    bps = 230400
    # 串口执行到这已经打开 再用open命令会报错
    mSerial = serial.Serial(portx, int(bps))
    if (mSerial.isOpen()):
        dat = 0xff;
        dat >> 2;
        print("open success")
        # 向端口些数据 字符串必须译码
        mSerial.write("hello".encode())
        mSerial.flushInput()  # 清空缓冲区
    else:
        print("open failed")
        serial.close()  # 关闭端口
    th1 = threading.Thread(target=Serial)
    th1.start()
    timer = pg.QtCore.QTimer()
    timer.timeout.connect(plotData)  # 定时刷新数据显示
    timer.start(1)  # 多少ms调用一次

    win2 = pg.GraphicsWindow()  # 建立窗口
    win2.setWindowTitle(u'channel 1 功率谱')
    win2.resize(350, 200)  # 小窗口大小
    p = win2.addPlot()  # 把图p加入到窗口中
    p.showGrid(x=True, y=True)  # 把X和Y的表格打开
    p.setRange(xRange=[0, 1024], yRange=[-100, 100], padding=0)
    curve2 = p.plot()  # 频谱
    win2.nextRow()
    p = win2.addPlot()  # 把图p加入到窗口中
    p.showGrid(x=True, y=True)  # 把X和Y的表格打开
    p.setRange(xRange=[0, 1024], yRange=[-100, 100], padding=0)
    curve3 = p.plot()  # 滤波后频谱

    win3 = pg.GraphicsWindow()  # 建立窗口
    win3.setWindowTitle(u'channel 1 滤波后sub:{},ges{},tri{}'.format(subject,gesture,trial))
    win3.resize(350, 200)  # 小窗口大小

    p = win3.addPlot()  # 把图p加入到窗口中
    p.showGrid(x=True, y=True)  # 把X和Y的表格打开
    p.setRange(xRange=[0, 1000], yRange=[0, 3.3], padding=0)
    p.setDownsampling(mode='peak')
    p.setClipToView(True)
    curve4 = p.plot()  # 频谱

    #proxy = pg.SignalProxy(p.scene().sigMouseMoved, rateLimit=60, slot=mouseMoved)  #调用鼠标函数，实现鼠标数据显示
    proxy = pg.SignalProxy(win.scene().sigMouseClicked, rateLimit=60, slot=mouseClicked)  #snapshot
    proxy2 = pg.SignalProxy(win2.scene().sigMouseClicked, rateLimit=60, slot=mouseClicked2)  #threshold
    proxy3 = pg.SignalProxy(win3.scene().sigMouseClicked, rateLimit=60, slot=mouseClicked3)  # increase gesture

    app.exec_()
