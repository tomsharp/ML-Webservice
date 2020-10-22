# helper functions
def compressed_pickle(path, data):
    with bz2.BZ2File(path, 'w') as f: 
        cPickle.dump(data, f)

def decompress_pickle(path):
    data = bz2.BZ2File(path, 'rb')
    data = cPickle.load(data)
    return data
