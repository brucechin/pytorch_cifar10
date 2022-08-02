

import numpy as np


file = open('REJAFADA.data', 'r')
label = []
features = []
while(True):
    data = file.readline().strip('\n')
    if data == '':
        break
    raw = data.split(',')
    if(raw[1] == 'B'):
        label.append(0)
    else:
        label.append(1)
    
    raw_feature = raw[2:]
    feature = np.array(list(map(int, raw_feature)))
    features.append(feature)


