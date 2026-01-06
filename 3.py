f = open("sample1.txt", "w")
f.write("Hello World\n")
f.write("Welcome to File Handling in Python\n")
f.close()

f1 = open("sample1.txt", "a")
f1.write("This is an appended line.\n")
f1.close()

