import os
import gzip
import pickle
from collections import OrderedDict


class StreamList:
    def __init__(self, file_path, counts, buffer_size=10240):
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.counts = counts
        if not os.path.exists(file_path):
            with gzip.open(file_path, 'wb') as f:
                pass
        self.cache = OrderedDict()
    
    def __iter__(self):
        with gzip.open(self.file_path, 'rb') as f:
            while True:
                try:
                    yield pickle.load(f)
                except EOFError:
                    break
    
    def append(self, item):
        with gzip.open(self.file_path, 'ab') as f:
            pickle.dump(item, f)
        self.counts += 1

    def extend(self, items):
        with gzip.open(self.file_path, 'ab') as f:
            for item in items:
                pickle.dump(item, f)
        self.counts += len(items)
    
    def __getitem__(self, index):
        with gzip.open(self.file_path, 'rb') as f:
            if isinstance(index, int):
                if index < 0:
                    index += len(self)
                if index < 0 or index >= len(self):
                    raise IndexError("Index out of range")
                if index in self.cache:
                    return self.cache[index]
                for i in range(index + 1):
                    try:
                        item = pickle.load(f)
                        self.__update_cache(i, item)
                    except EOFError:
                        raise IndexError("Index out of range")
                return item
            elif isinstance(index, slice):
                indices = range(*index.indices(len(self)))
                values = {i: self.cache[i] for i in indices if i in self.cache}
                missing = set(indices) - values.keys()
                # Consume gzip records sequentially even for cached/skipped indices.
                # Retain selected records and restore the requested slice order.
                for i in range(max(missing, default=-1) + 1):
                    try:
                        item = pickle.load(f)
                        if i in missing:
                            values[i] = item
                            self.__update_cache(i, item)
                    except EOFError:
                        raise IndexError("Stream ended before the declared record count") from None
                return [values[i] for i in indices]
            else:
                raise TypeError("Invalid index type")
    
    def __update_cache(self, key, item):
        if len(self.cache) >= self.buffer_size:
            self.cache.popitem(last=False)
        self.cache[key] = item

    def __len__(self):
        return self.counts

    def delete(self):
        os.remove(self.file_path)
    