import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MovingAverageFilter:
    def __init__(self, y_initial_measure, num_average=2.0):
        self.y_estimate = y_initial_measure
        # Code
        self.y_estimate = y_initial_measure
        self.num_average = int(num_average)  # 이동 평균 개수
        self.history = [y_initial_measure]   # 최근 측정값 저장용 리스트
    def estimate(self, y_measure):
        # Code
        self.history.append(y_measure)
        if len(self.history) > self.num_average:
            self.history.pop(0)  # 가장 오래된 값 제거
        self.y_estimate = np.mean(self.history)
    
if __name__ == "__main__":
    #signal = pd.read_csv("week_01_filter/Data/example_Filter_1.csv")      
    #signal = pd.read_csv("week_01_filter/Data/example_Filter_2.csv")
    signal = pd.read_csv("D:/IVS 3기/만클 현직자 프로젝트/99_Release/01_Filter/Data/example_Filter_2.csv")

    y_estimate = MovingAverageFilter(signal.y_measure[0])
    for i, row in signal.iterrows():
        y_estimate.estimate(signal.y_measure[i])
        signal.y_estimate[i] = y_estimate.y_estimate

    plt.figure()
    plt.plot(signal.time, signal.y_measure,'k.',label = "Measure")
    plt.plot(signal.time, signal.y_estimate,'r-',label = "Estimate")
    plt.xlabel('time (s)')
    plt.ylabel('signal')
    plt.legend(loc="best")
    plt.axis("equal")
    plt.grid(True)
    plt.show()



