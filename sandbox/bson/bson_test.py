import bson

save_path = 'results/julia_bob.bson'

with open(save_path,'rb') as f:
    data = bson.BSON.decode(f.read())

print(data)
