import json

dic = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

with open("a.json", "w") as f:
    json.dump(dic, f)

# nosql mongodb