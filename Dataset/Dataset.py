root = 'data.txt'

data = open(root,'r')
train = open('Iris_train.txt','w')
test = open('Iris_test.txt','w')
val = open('Iris_val.txt','w')
Flower_name_1 = 'Iris-setosa\n'
Flower_name_2 = 'Iris-versicolor\n'
Flower_name_3 = 'Iris-virginica\n'
Flower_name = '0'
count = 0
for i in data:
    transform = i.split(',')
    if transform[4] == Flower_name_1:
        Flower_name = '0'
    if transform[4] == Flower_name_2:
        Flower_name = '1'
    if transform[4] == Flower_name_3:
        Flower_name = '2'
    if count%15==0:
        tester = (transform[0]+','+transform[1]+','+transform[2]+','+transform[3]+' '+Flower_name+'\n')
        print(tester)
        test.write(tester)
    elif count%14==0 and count != 0:
        valer = (transform[0] + ',' + transform[1] + ',' + transform[2] + ',' + transform[3] + ' ' + Flower_name + '\n')
        print(valer)
        val.write(valer)
    else:
        trainer = (transform[0]+','+transform[1]+','+transform[2]+','+transform[3]+' '+Flower_name+'\n')
        print(trainer)
        train.write(trainer)
    count = count+1
data.close()
train.close()
test.close()
val.close()