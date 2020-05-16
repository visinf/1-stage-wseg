
class StatManager(object):

    def __init__(self):
        self.func_keys = {}
        self.vals = {}
        self.vals_count = {}
        self.formats = {}

    def reset(self):
        for k in self.vals:
            self.vals[k] = 0.0
            self.vals_count[k] = 0.0

    def add_val(self, key, form="{:4.3f}"):
        self.vals[key] = 0.0
        self.vals_count[key] = 0.0
        self.formats[key] = form

    def get_val(self, key):
        return self.vals[key], self.vals_count[key]

    def add_compute(self, key, func, form="{:4.3f}"):
        self.func_keys[key] = func
        self.add_val(key)
        self.formats[key] = form
    
    def update_stats(self, key, val, count = 1):
        if not key in self.vals:
            self.add_val(key)

        self.vals[key] += val
        self.vals_count[key] += count

    def compute_stats(self, a, b, size = 1):

        for k, func in self.func_keys.iteritems():
            self.vals[k] += func(a, b)
            self.vals_count[k] += size

    def has_vals(self, k):
        if not k in self.vals_count:
            return False
        return self.vals_count[k] > 0

    def summarize_key(self, k):
        if self.has_vals(k):
            return self.vals[k] / self.vals_count[k]
        else:
            return 0

    def summarize(self, epoch = 0, verbose = True):

        if verbose:
            out = "\tEpoch[{:03d}]".format(epoch)
            for k in self.vals:
                if self.has_vals(k):
                    out += (" / {} " + self.formats[k]).format(k, self.summarize_key(k))
            print(out)
