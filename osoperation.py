import os

# os.mkdir("amar")

# os.rename("amar", "anil")

# os.mkdir("anil/kumar.txt")

print(os.listdir("."))
print("Current Working Directory:", os.getcwd())
print(os.path.exists("anil/kumar.txt"))


print(os.path.isfile("anil/kumar.txt"))
print(os.path.isdir("anil/kumar.txt"))
print(os.path.splitext("anil/kumar.txt"))
print(os.path.join("anil", "kumar.txt"))
print(os.path.basename("anil/kumar.txt"))
print(os.path.dirname("anil/kumar.txt"))
print(os.path.getsize("anil/kumar.txt"))
print(os.path.abspath("anil/kumar.txt"))
print(os.path.getctime("anil/kumar.txt"))
print(os.path.getmtime("anil/kumar.txt"))
print(os.path.getatime("anil/kumar.txt"))
print(os.path.samefile("anil/kumar.txt", "anil/kumar.txt"))


import shutil
shutil.copy("anil/kumar.txt", "anil/kumar_copy.txt")
shutil.move("anil/kumar_copy.txt", "anil/kumar_moved.txt")
shutil.rmtree("anil")
shutil.make_archive("anil_archive", 'zip', "anil")
shutil.unpack_archive("anil_archive.zip", "anil_unpacked", 'zip')
