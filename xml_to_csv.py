import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    train_list = []
    eval_list = []
    i = 0
    # 读取注释文件
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
       # for member in root.findall('object'):
        value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     root.find('object')[0].text,
                     int(root.find('object')[4][0].text),
                     int(root.find('object')[4][1].text),
                     int(root.find('object')[4][2].text),
                     int(root.find('object')[4][3].text)
                     )
# 将所有数据分为样本集和验证集，一般按照3:1的比例
        if i%4==3:
            eval_list.append(value)
        else:
            train_list.append(value)
        i += 1
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']

    # 保存为CSV格式
    train_df = pd.DataFrame(train_list, columns=column_name)
    eval_df = pd.DataFrame(eval_list, columns=column_name)
    train_df.to_csv('D:\\Images\\data_train.csv',encoding='utf_8_sig',index=None)
    eval_df.to_csv('D:\\Images\\data_eval.csv',encoding='utf_8_sig', index=None)



path = 'D:\\Images\\Annotations'
xml_to_csv(path)
print('Successfully converted xml to csv.')
