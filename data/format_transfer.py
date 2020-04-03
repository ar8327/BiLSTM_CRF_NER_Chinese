import re
import jieba
import config

C = config.Config()
jieba.load_userdict('coal_dict.txt')

class dataLoader():
    def __init__(self,origin_file_name):
        self.fn = origin_file_name
        self.data = open(self.fn,'r')
        self.tag_list = set()
        self.tgt_output = open(C.traindata,'w')



    def readline(self,line):
        labels = []
        pattern = re.compile('\{[^}]*\}}')
        str = line
        result = re.findall(pattern,str)
        for r in result:
            line = line.replace(r,'_LABELED_')
        line = line.replace(' ', '').replace('\n', '')
        seg_list = jieba.cut(line)
        real_seg_list = []
        count = 0
        for word in seg_list:
            if word == '_LABELED_':
                labeled_data = result[count][2:-2]
                try:
                    tag,data = labeled_data.split(':')
                    data = data.replace(' ', '').replace('\n', '')
                except:
                    count += 1
                    continue
                data = list(jieba.cut(data))
                for d in range(len(data)):
                    if d == 0:
                        labels.append('B-'+tag)
                        self.tag_list.update('B-'+tag)
                    elif d != 0 and d == len(data) - 1:
                        labels.append('E-'+tag)
                        self.tag_list.update('E-'+tag)
                    else:
                        labels.append('M-'+tag)
                        self.tag_list.update('M-'+tag)
                    real_seg_list.append(data[d])
                count += 1
            else:
                real_seg_list.append(word)
                labels.append('O')
                self.tag_list.update('O')
        return real_seg_list,labels

    def generateMyFormat(self):
        for line in self.data.readlines():
            X,Y = self.readline(line)
            for i in range(len(X)):
                self.tgt_output.write(X[i]+" "+Y[i]+"\n")
            self.tgt_output.write("\n")
        self.tgt_output.close()


if __name__ == '__main__':
    dataLoader = dataLoader('origindata.txt')
    dataLoader.generateMyFormat()