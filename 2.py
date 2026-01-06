
with open('sample.txt', 'r') as file:
    content = file.read()
    print(content)


f = open('sample.txt', 'r')
data = f.read()
print(data)
f.close()


f1 = open('sample.txt', 'r')
data1 = f1.readlines()
print(data1)
f1.close()


for i in data1:
    print(i)


print("Hello")