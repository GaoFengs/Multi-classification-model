import os
import csv

path = 'D:\pyWork\dogsVScats-master\\result'
result_list = os.listdir(path)
my_result = {}
print(result_list)

count = len(result_list)
f = []
for k in range(count):
    f.append(csv.reader(open(path + '\\' + result_list[k],'r')))

for i in range(1000):
    my_result[str(i + 1)] =[0,0]
# AD = 0 CN = 1
k = 1
for one_f in f:

    print('第'+str(i)+'个文件。')
    for data in one_f:
        if data[0] == 'uuid':
            continue
        if data[1] == 'AD':
            my_result[data[0]][0] += 1
        else:
            my_result[data[0]][1] += 1
    i += 1

print(my_result)
f = open('my_result.csv', 'a', newline='', encoding='utf-8')
writer = csv.writer(f)
writer.writerow(['uuid', 'label'])
f.close()
print('创建文件成功！')
for i in my_result:
    if my_result[i][0] > my_result[i][1]:
        f = open('my_result.csv', 'a', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow([i, 'AD'])
        f.close()
        print('写入文件成功！')
    else:
        f = open('my_result.csv', 'a', newline='', encoding='utf-8')
        writer = csv.writer(f)
        writer.writerow([i, 'CN'])
        f.close()
        print('写入文件成功！')





