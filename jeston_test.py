import torch
import torch.nn as nn
import array
import threading
import numpy as np
from queue import Queue

from PyQt5.QtWidgets import QDesktopWidget
from scipy import signal
import time
from scipy.fftpack import fft
import scipy.io as sio
import serial
from collections import Counter
import pyqtgraph as pg

i = 0
q = Queue(maxsize=0)

VOTE_NUM = 10
SAVE_LEN = 5000
num_classes = 4
channelNum = 8
historyLength = 1000


gesture_list1 = ['手势向外', '手势向内', '手势向左', '手势向右']
gesture_list = ['<img src=1.png>',
                '<img src=2.png>',
                '<img src=3.png>',
                '<img src=4.png>']

global F1
global M;
global lamda;
global I;
global c;
global x_noise;

global P_last;
global w_last;
global is_draw;
global label;

current_label = 0
count_plot = 0

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
M = 4  # %定义FIR滤波器阶数
lamda = 0.9  # 定义遗忘因子
I = np.eye(M)  # 生成对应的单位矩阵
c = 0.01  # 小正数 保证矩阵P非奇异
t = np.arange(1000) / 1000
t = t.reshape(1000, 1)

x_noise = np.hstack(
    (np.sin(2 * np.pi * F1 * t), np.cos(2 * np.pi * F1 * t), np.sin(2 * np.pi * F2 * t), np.cos(2 * np.pi * F2 * t),
     # np.sin(2*np.pi*F3*t),np.cos(2*np.pi*F3*t),np.sin(2*np.pi*F4*t),np.cos(2*np.pi*F4*t),
     # np.sin(2*np.pi*F5*t),np.cos(2*np.pi*F5*t),np.sin(2*np.pi*F6*t),np.cos(2*np.pi*F6*t),
     # np.sin(2*np.pi*F7*t),np.cos(2*np.pi*F7*t),np.sin(2*np.pi*F8*t),np.cos(2*np.pi*F8*t),
     # np.sin(2*np.pi*F9*t),np.cos(2*np.pi*F9*t),np.sin(2*np.pi*F10*t),np.cos(2*np.pi*F10*t)
     ));
P_last = I / c;
w_last = np.zeros([M, 1]);

b, a = signal.butter(8, 2 * 10 / 1000, 'highpass')  # filter direct ciruit


# Convolutional neural network part of VGGNet
class JESTON_MODEL(nn.Module):
    def __init__(self, used_gpu, nclass, dropout, output_weight=False, init_weights=False):
        super(JESTON_MODEL, self).__init__()
        self.nclass = nclass
        self.output_weight = output_weight
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.softmax = nn.Softmax(dim=-1)

        self.pooling = nn.MaxPool1d(kernel_size=3, stride=3, padding=1)

        self.bn_16 = nn.BatchNorm1d(16)
        self.bn_32 = nn.BatchNorm1d(32)
        self.bn_64_3 = nn.BatchNorm1d(64)
        self.bn_64_4 = nn.BatchNorm1d(64)
        self.bn_64_5 = nn.BatchNorm1d(64)
        self.bn_128 = nn.BatchNorm1d(128)
        self.bn_256 = nn.BatchNorm1d(256)
        self.bn_512 = nn.BatchNorm1d(512)

        self.layer_8_16 = nn.Conv1d(channelNum, 16, kernel_size=3, padding=1)
        self.layer_8_64 = nn.Conv1d(channelNum, 64, kernel_size=3, padding=1)
        self.layer_16_32 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.layer_32_64 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.layer_64_64 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.layer_64_128 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.layer_128_128 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.layer_128_256 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.layer_256_256 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.layer_256_512 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.layer_512_512 = nn.Conv1d(512, 512, kernel_size=3, padding=1)

        self.layer_256_256_2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_2 = nn.BatchNorm1d(256)
        self.layer_256_256_3 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_3 = nn.BatchNorm1d(256)
        self.layer_256_256_4 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.bn_256_4 = nn.BatchNorm1d(256)
        self.layer_256_512_5 = nn.Conv1d(256, 512, kernel_size=3, padding=1)
        self.bn_512_5 = nn.BatchNorm1d(512)
        self.layer_512_512_6 = nn.Conv1d(512, 512, kernel_size=3, padding=1)
        self.bn_512_6 = nn.BatchNorm1d(512)

        self.classifier2 = nn.Sequential(nn.Linear(64, nclass))

    def forward(self, x):
        # Covolutions

        x = self.layer_8_16(x)
        x = self.bn_16(x)
        x = self.relu(x)
        x = self.layer_16_32(x)
        x = self.bn_32(x)
        x = self.relu(x)
        x = self.layer_32_64(x)
        x = self.bn_64_3(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.layer_64_64(x)
        x = self.bn_64_4(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.layer_64_64(x)
        # x = self.dropout(x)
        x = self.bn_64_5(x)
        x = self.relu(x)

        '''
        x = self.layer_256_256(x)
        x = self.bn_256(x)
        x = self.relu(x)
        x = self.pooling(x)

        x = self.layer_256_512(x)
        x = self.bn_512(x)
        x = self.relu(x)
        x = self.layer_512_512(x)
        x = self.bn_512(x)
        x = self.relu(x)
        '''

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier2(x)
        return x


if torch.cuda.is_available():
    print('use gpu')
    device = 'cuda:0'
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
    model = JESTON_MODEL(used_gpu=[0], nclass=num_classes, dropout=0.5)
    model = torch.nn.parallel.DataParallel(model, device_ids=[0])

else:
    print('use cpu')
    device = 'cpu'
    model = JESTON_MODEL(used_gpu=None, nclass=num_classes, dropout=0.5)

print('load model')
model.load_state_dict(torch.load('model.ckpt'))
print('load model finish')
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)


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
        d = sample[i];  # 输入新的期望信号
        x = x_noise[i, :].reshape(M, 1);  # 输入新的信号矢量
        tmp = np.dot(x.T, P_last)
        tmp = np.dot(tmp, x)

        # if i == 0: ###### do not remove
        #    print(lamda,tmp,lamda+tmp)

        K = (np.dot(P_last, x)) / (lamda + tmp);  # 计算增益矢量
        # K = (np.dot(P_last , x))/(lamda + x.T *  P_last * x);   #计算增益矢量
        y = np.dot(-x.T, w_last);  # 计算FIR滤波器输出

        Eta = (d + y);  # 计算估计的误差
        w = w_last + K * Eta;  # 计算滤波器系数矢量
        P = (I - K * x.T) * P_last / lamda;  # 计算误差相关矩阵
        P_last = P;
        w_last = w;
        sample1[i] = Eta;
    return sample1


def Serial():
    global i
    global q
    global current_label
    head = [-1, -1, -1, -1, -1]
    count = 0
    data2 = array.array('i')  # 可动态改变数组的大小,double型数组
    data2 = np.zeros((channelNum, historyLength)).__array__('d')  # 把数组长度定下来
    data_rls2 = array.array('i')  # 可动态改变数组的大小,double型数组
    data_rls2 = np.zeros((channelNum, historyLength)).__array__('d')  # 把数组长度定下来
    vote_queue = [0 for i in range(VOTE_NUM)]
    vote_barrel = [0 for i in range(num_classes)]

    while (True):
        for i in range(5):
            head[i] = mSerial.read(1)
        while True:
            if head[0] == b'\x01' and head[1] == b'\x02' and head[2] == b'\x03' and head[3] == b'\x04'\
                    and head[4] == b'\x05':

                count = count + 1
                print("rev:", count)
                message = mSerial.read(320)
                emg_serial_data = np.array([x for x in message]).reshape(8, 40) / 4096 * 3.3
                emg_frame_data = emg_serial_data[:, 0::2] * 256 + emg_serial_data[:, 1::2]

                data[:, 0:980] = data2[:, 20:1000]
                data[:, 980:1000] = emg_frame_data
                data2 = data

                model_data = data[:,940:1000]
                # for i in range(channelNum):
                data_rls[0, 0:980] = data_rls2[0, 20:1000]
                data_rls[0, 980:1000] = rls_algo(data[0, 980:1000])
                data_rls2 = data_rls

                images = torch.Tensor(model_data).reshape(1, 8, 60)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                print(predicted)
                vote_queue[0:len(vote_queue) - 1] = vote_queue[1:len(vote_queue)]
                vote_queue[len(vote_queue) - 1] = predicted[0].item()

                vote_result = Counter(vote_queue).most_common(1)[0][0]
                current_label = vote_result
                print(vote_queue)
                print(Counter(vote_queue))
                print(vote_result)

                for i in range(5):
                    head[i] = mSerial.read(1)
            else:
                head[:-1] = head[1:]
                head[4] = mSerial.read(1)


def plotData():
    global current_label;
    global count_plot;

    count_plot = count_plot + 1
    print('plot:', count_plot)
    for channel in range(len(curves)):
        curves[channel].setData(data[channel])
    # 图片label内容设定
    # label.setText(gesture_list[current_label])
    # 文字label内容设定
    label1.setText(gesture_list1[current_label])


if __name__ == "__main__":

    app = pg.mkQApp()  # 建立app
    win = pg.GraphicsWindow()  # 建立窗口
    win.setWindowTitle(u'肌电8通道波形图')
    win.resize(1200, 900)  # 小窗口大小
    screen = QDesktopWidget().screenGeometry()
    size = win.geometry()
    win.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    data = array.array('i')  # 可动态改变数组的大小,double型数组
    a = 0
    data = np.zeros((channelNum, historyLength)).__array__('d')  # 把数组长度定下来

    data_rls = array.array('i')  # 可动态改变数组的大小,double型数组
    data_rls = np.zeros((channelNum, historyLength)).__array__('d')  # 把数组长度定下来

    curves = []
    # label=win.addLabel(gesture_list[current_label])
    # win.nextRow()
    label1=win.addLabel(gesture_list1[current_label])
    win.nextRow()

    for channel in range(channelNum):
        p = win.addPlot()  # 把图p加入到窗口中
        p.showGrid(x=True, y=True)  # 把X和Y的表格打开
        p.setRange(xRange=[0, historyLength], yRange=[0, 3.3], padding=0)
        p.setLabel(axis='left', text='y / V')  # 靠左

        curve = p.plot()  # 绘制一个图形
        curve.setData(data[i])
        curves.append(curve)

        win.nextRow()

    # portx = 'COM5'
    # portx = '/dev/tty.usbserial-AR0K409Y'
    # portx = "/dev/ttyTHS1"
    portx = '/dev/tty.usbmodemADPT0157431'

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
    timer.start(500)  # 多少ms调用一次   
    app.exec_()
