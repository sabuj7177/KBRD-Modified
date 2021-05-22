# import requests
# import json
#
# url = "http://localhost:4999/processQuery"
# payload = {"nlquery":"Back to Hannibal: The Return of Tom Sawyer and Huckleberry Finn film"}
# res = requests.post(url, json=payload, headers = {"Content-Type": "application/json"})
# log_file = open("log.txt", "a")
# print(res.json())
# log_file.write(json.dumps(res.json())+"\n")
# log_file.write(json.dumps(res.json())+"\n")
#
# print(len(res.json()["rerankedlists"]["0"]))
# print(res.json()["rerankedlists"]["0"][0][1])
# log_file.close()
# from Levenshtein import distance as levenshtein_distance
#
# movie_name = "The Twilight Saga"
# dist = levenshtein_distance(movie_name, "The Twilight")
# print((dist*100)/len(movie_name))

def get_movie_name_from_entity(entity):
    frost_nixon = "http://dbpedia.org/resource/Frost/Nixon_(film)"
    farenheit = "http://dbpedia.org/resource/Fahrenheit_9/11"
    if (entity == frost_nixon) or (entity == farenheit):
        x = find_nth_overlapping(entity[::-1], "/", 2)
    else:
        x = entity[::-1].find("/")
    movie = entity[-x:]
    return movie.replace("_", " ")

def find_nth_overlapping(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+1)
        n -= 1
    return start

print(get_movie_name_from_entity("http://dbpedia.org/resource/Frost/Nixon_(film)"))
print(get_movie_name_from_entity("http://dbpedia.org/resource/Fahrenheit_9/11"))