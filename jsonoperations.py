import json

dic = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

with open("a.json", "w") as f:
    json.dump(dic, f)

# nosql mongodb

f = open("a.json", "r")
data = json.load(f)
print(data)

print(data['name'])
f.close()