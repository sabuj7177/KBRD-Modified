import csv
import json
import os
import pickle as pkl
import random
import re
from collections import defaultdict

import parlai.core.build_data as build_data
from parlai.scripts.train_model import setup_args
import requests
from Levenshtein import distance as levenshtein_distance


def _split_data(redial_path):
    # Copied from https://github.com/RaymondLi0/conversational-recommendations/blob/master/scripts/split-redial.py
    data = []
    for line in open(os.path.join(redial_path, "train_data.jsonl"), encoding="utf-8"):
        data.append(json.loads(line))
    random.shuffle(data)
    n_data = len(data)
    split_data = [data[: int(0.9 * n_data)], data[int(0.9 * n_data):]]

    with open(os.path.join(redial_path, "train_data.jsonl"), "w", encoding="utf-8") as outfile:
        for example in split_data[0]:
            json.dump(example, outfile)
            outfile.write("\n")
    with open(os.path.join(redial_path, "valid_data.jsonl"), "w", encoding="utf-8") as outfile:
        for example in split_data[1]:
            json.dump(example, outfile)
            outfile.write("\n")


def _entity2movie(entity, abstract=""):
    # strip url
    x = entity[::-1].find("/")
    movie = entity[-x:-1]
    movie = movie.replace("_", " ")

    # extract year
    pattern = re.compile(r"\d{4}")
    match = re.findall(pattern, movie)
    year = match[0] if match else None
    # if not find in entity title, find in abstract
    if year is None:
        pattern = re.compile(r"\d{4}")
        match = re.findall(pattern, abstract)
        if match and 1900 < int(match[0]) and int(match[0]) < 2020:
            year = match[0]

    # recognize (20xx film) or (film) to help disambiguation
    pattern = re.compile(r"\(.*film.*\)")
    match = re.findall(pattern, movie)
    definitely_is_a_film = match != []

    # remove parentheses
    while True:
        pattern = re.compile(r"(.+)( \(.*\))")
        match = re.search(pattern, movie)
        if match:
            movie = match.group(1)
        else:
            break
    movie = movie.strip()

    return movie, year, definitely_is_a_film


DBPEDIA_ABSTRACT_PATH = "dbpedia/short_abstracts_en.ttl"
DBPEDIA_PATH = "dbpedia/mappingbased_objects_en.ttl"


def _build_dbpedia(dbpedia_path):
    movie2entity = {}
    movie2years = defaultdict(set)
    with open(dbpedia_path, encoding="utf-8") as f:
        for line in f.readlines():
            if line.startswith("#"):
                continue
            entity, line = line[: line.index(" ")], line[line.index(" ") + 1:]
            _, line = line[: line.index(" ")], line[line.index(" ") + 1:]
            abstract = line[:-4]
            movie, year, definitely_is_a_film = _entity2movie(entity, abstract)
            if (movie, year) not in movie2entity or definitely_is_a_film:
                movie2years[movie].add(year)
                movie2entity[(movie, year)] = entity
    return {"movie2years": movie2years, "movie2entity": movie2entity}


def _load_kg(path):
    kg = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for line in f.readlines():
            tuples = line.split()
            if tuples and len(tuples) == 4 and tuples[-1] == ".":
                h, r, t = tuples[:3]
                # TODO: include property/publisher and subject/year, etc
                if "ontology" in r:
                    kg[h].append((r, t))
    return kg


def _extract_subkg(kg, seed_set, n_hop, log_file):
    print("Seed num " + str(len(seed_set)))
    log_file.write("Seed num " + str(len(seed_set)) + "\n")
    subkg = defaultdict(list)
    subkg_hrt = set()

    ripple_set = []
    for h in range(n_hop):
        memories_h = []
        memories_r = []
        memories_t = []

        if h == 0:
            tails_of_last_hop = seed_set
        else:
            tails_of_last_hop = ripple_set[-1][2]

        for entity in tails_of_last_hop:
            for tail_and_relation in kg[entity]:
                h, r, t = entity, tail_and_relation[0], tail_and_relation[1]
                if (h, r, t) not in subkg_hrt:
                    subkg[h].append((r, t))
                    subkg_hrt.add((h, r, t))
                memories_h.append(h)
                memories_r.append(r)
                memories_t.append(t)

        ripple_set.append((memories_h, memories_r, memories_t))

    return subkg


def build(opt):
    # get path to data directory
    dpath = os.path.join(opt["datapath"], "redial")
    # define version if any
    version = None
    log_file = open(os.path.join(dpath, "log.txt"), "a")

    # check if data had been previously built
    # if not build_data.built(dpath, version_string=version):
    if True:
        print("[building data: " + dpath + "]")

        # make a clean directory if needed
        if build_data.built(dpath):
            # an older version exists, so remove these outdated files.
            build_data.remove_dir(dpath)
        build_data.make_dir(dpath)

        # download the data.
        fname = "redial_dataset.zip"
        url = "https://github.com/ReDialData/website/raw/data/" + fname  # dataset URL
        build_data.download(url, dpath, fname)

        # uncompress it
        build_data.untar(dpath, fname)

        _split_data(dpath)

        dbpedia = _build_dbpedia(DBPEDIA_ABSTRACT_PATH)
        movie2entity = dbpedia["movie2entity"]
        movie2years = dbpedia["movie2years"]

        # print(movie2entity)
        kg = _load_kg(DBPEDIA_PATH)

        # Match REDIAL movies to dbpedia entities
        movies_with_mentions_path = os.path.join(dpath, "movies_with_mentions.csv")
        with open(movies_with_mentions_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            id2movie = {int(row[0]): row[1] for row in reader if row[0] != "movieId"}
        id2entity = {}
        count = 0
        previous_non_matched = 0
        for movie_id in id2movie:
            movie = id2movie[movie_id]
            pattern = re.compile(r"(.+)\((\d+)\)")
            match = re.search(pattern, movie)
            if match is not None:
                name, year = match.group(1).strip(), match.group(2)
            else:
                name, year = movie.strip(), None
            if year is not None:
                if (name, year) in movie2entity:
                    id2entity[movie_id] = movie2entity[(name, year)]
                else:
                    if len(movie2years) == 1:
                        id2entity[movie_id] = movie2entity[(name, movie2years[name][0])]
                    else:
                        # id2entity[movie_id] = None
                        previous_non_matched += 1
                        ent = get_entity(name, kg, log_file)
                        print("Final output")
                        print(ent)
                        log_file.write("Final output"+"\n")
                        if ent is None:
                            log_file.write("None")
                        else:
                            log_file.write(ent)
                        log_file.write("\n")
                        id2entity[movie_id] = ent
                        if ent is None:
                            count += 1

            else:
                if (name, year) in movie2entity:
                    id2entity[movie_id] = movie2entity[(name, year)]
                else:
                    # id2entity[movie_id] = None
                    previous_non_matched += 1
                    ent = get_entity(name, kg, log_file)
                    print("Final output")
                    print(ent)
                    log_file.write("Final output"+"\n")
                    if ent is None:
                        log_file.write("None")
                    else:
                        log_file.write(ent)
                    log_file.write("\n")
                    id2entity[movie_id] = ent
                    if ent is None:
                        count += 1
                #     id2entity[movie_id] = (
                #     movie2entity[(name, year)] if (name, year) in movie2entity else None
                # )
        # HACK: make sure movies are matched to different entities
        matched_entities = set()
        for movie_id in id2entity:
            if id2entity[movie_id] is not None:
                if id2entity[movie_id] not in matched_entities:
                    matched_entities.add(id2entity[movie_id])
                else:
                    previous_non_matched += 1
                    movie = id2movie[movie_id]
                    pattern = re.compile(r"(.+)\((\d+)\)")
                    match = re.search(pattern, movie)
                    if match is not None:
                        name, year = match.group(1).strip(), match.group(2)
                    else:
                        name, year = movie.strip(), None
                    # print("In hack: movie name: " + name)
                    ent = get_entity(name, kg, log_file)
                    print("Final output")
                    print(ent)
                    log_file.write("Final output"+"\n")
                    if ent is None:
                        log_file.write("None")
                    else:
                        log_file.write(ent)
                    log_file.write("\n")
                    if ent is None:
                        id2entity[movie_id] = None
                        count += 1
                    else:
                        if ent not in matched_entities:
                            matched_entities.add(ent)
                            id2entity[movie_id] = ent
                            # print("Inserting entity")
                            # print(ent)
                        else:
                            id2entity[movie_id] = None
                            count += 1

        print("Not matched movies: " + str(count))
        print("Previous Not matched movies: " + str(previous_non_matched))
        log_file.write("Not matched movies: " + str(count) + "\n")
        log_file.write("Previous Not matched movies: " + str(previous_non_matched) + "\n")

        # Extract sub-kg related to movies

        subkg = _extract_subkg(
            kg,
            [
                id2entity[k]
                for k in id2entity
                if id2entity[k] is not None and kg[id2entity[k]] != []
            ],
            2,
            log_file
        )

        not_empty_id = 0
        for movie_id in id2entity:
            if id2entity[movie_id] is not None:
                subkg[id2entity[movie_id]].append(('self_loop', id2entity[movie_id]))
                not_empty_id += 1
            else:
                subkg[movie_id].append(('self_loop', movie_id))
        print("Not empty id num "+str(not_empty_id))
        log_file.write("Not empty id num "+str(not_empty_id) + "\n")
        entities = set([k for k in subkg]) | set([x[1] for k in subkg for x in subkg[k]])
        entity2entityId = dict([(k, i) for i, k in enumerate(entities)])
        relations = set([x[0] for k in subkg for x in subkg[k]])
        relation2relationId = dict([(k, i) for i, k in enumerate(relations)])
        subkg_idx = defaultdict(list)
        for h in subkg:
            for r, t in subkg[h]:
                subkg_idx[entity2entityId[h]].append((relation2relationId[r], entity2entityId[t]))
        movie_ids = []
        for k in id2entity:
            movie_ids.append(entity2entityId[id2entity[k]] if id2entity[k] is not None else entity2entityId[k])

        pkl.dump(id2entity, open(os.path.join(dpath, "id2entity.pkl"), "wb"))
        pkl.dump(dbpedia, open(os.path.join(dpath, "dbpedia.pkl"), "wb"))
        pkl.dump(subkg_idx, open(os.path.join(dpath, "subkg.pkl"), "wb"))
        pkl.dump(entity2entityId, open(os.path.join(dpath, "entity2entityId.pkl"), "wb"))
        pkl.dump(relation2relationId, open(os.path.join(dpath, "relation2relationId.pkl"), "wb"))
        pkl.dump(movie_ids, open(os.path.join(dpath, "movie_ids.pkl"), "wb"))

        # mark the data as built
        build_data.mark_done(dpath, version_string=version)
        log_file.close()


def get_entity(movie_name, kg, log_file):
    print("###############################################")
    print(movie_name)
    log_file.write("###############################################"+"\n")
    log_file.write(movie_name+"\n")
    if len(movie_name) <= 0:
        return None
    threshold_dist_with_film = 10
    threshold_dist_without_film = 10
    url = "http://localhost:4999/processQuery"
    payload_without_film = {"nlquery": movie_name}
    res_without_film = requests.post(url, json=payload_without_film, headers={"Content-Type": "application/json"})
    print(res_without_film.json())
    log_file.write(json.dumps(res_without_film.json()))
    log_file.write("\n")
    payload_with_film = {"nlquery": movie_name+" film"}
    res_with_film = requests.post(url, json=payload_with_film, headers={"Content-Type": "application/json"})
    print(res_with_film.json())
    log_file.write(json.dumps(res_with_film.json()))
    log_file.write("\n")
    entities_without_film = []
    keys = ["0", "1", "2", "3", "4"]
    if "rerankedlists" in res_without_film.json():
        for key in keys:
            if key in res_without_film.json()["rerankedlists"]:
                if len(res_without_film.json()["rerankedlists"][key]) > 0:
                    entities_list = res_without_film.json()["rerankedlists"][key]
                    list_len = len(entities_list)
                    for i in range(list_len):
                        # generated_ent = "<" + entities_list[i][1] + ">"
                        # if kg[generated_ent] != []:
                        entities_without_film.append(entities_list[i][1])

    entities_with_film = []
    if "rerankedlists" in res_with_film.json():
        for key in keys:
            if key in res_with_film.json()["rerankedlists"]:
                if len(res_with_film.json()["rerankedlists"][key]) > 0:
                    entities_list = res_with_film.json()["rerankedlists"][key]
                    list_len = len(entities_list)
                    for i in range(list_len):
                        # generated_ent = "<" + entities_list[i][1] + ">"
                        # if kg[generated_ent] != []:
                        entities_with_film.append(entities_list[i][1])

    for ent in entities_with_film:
        if ent.endswith("_(film)"):
            movie_name_extracted = get_movie_name_from_entity(ent[:-7])
            dist = levenshtein_distance(movie_name_extracted, movie_name)
            dist_percentage = (dist * 100) / len(movie_name)
            if dist_percentage <= threshold_dist_with_film:
                return "<" + ent + ">"
        else:
            entities_without_film.append(ent)

    lowest_dist = 100000000
    lowest_ent = None
    for ent in entities_without_film:
        movie_name_extracted = get_movie_name_from_entity(ent)
        dist = levenshtein_distance(movie_name_extracted, movie_name)
        dist_percentage = (dist*100)/len(movie_name)
        if dist_percentage <= threshold_dist_without_film and dist_percentage < lowest_dist:
            lowest_dist = dist_percentage
            lowest_ent = "<" + ent + ">"

    return lowest_ent


def get_movie_name_from_entity(entity):
    frost_nixon = "http://dbpedia.org/resource/Frost/Nixon"
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


if __name__ == "__main__":
    parser = setup_args()
    parser.set_defaults(
        task="redial",
        dict_tokenizer="split",
        model="kbrd",
        dict_file="saved/tmp",
        model_file="saved/kbrd",
        fp16=True,
        batchsize=256,
        n_entity=64368,
        n_relation=214,
        # validation_metric="recall@50",
        validation_metric="base_loss",
        validation_metric_mode='min',
        validation_every_n_secs=30,
        validation_patience=5,
        tensorboard_log=True,
        tensorboard_tag="task,model,batchsize,dim,learningrate,model_file",
        tensorboard_metrics="loss,base_loss,kge_loss,l2_loss,acc,auc,recall@1,recall@10,recall@50",
    )
    opt = parser.parse_args()
    build(opt)
