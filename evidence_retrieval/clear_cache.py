import pickle

cache_file = "data/CIKM/cache/wiki_search_results.pkl"

with open(cache_file, 'rb') as f:
    cache = pickle.load(f)

before = len(cache)
cache = {k: v for k, v in cache.items() if v}
after = len(cache)

with open(cache_file, 'wb') as f:
    pickle.dump(cache, f)

print(f"Removed {before - after} empty entries, {after} entries remaining")