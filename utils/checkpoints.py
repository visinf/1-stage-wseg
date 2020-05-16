import os
import torch


class Checkpoint(object):

    def __init__(self, path, max_n=3):
        self.path = path
        self.max_n = max_n
        self.models = {}
        self.checkpoints = []

    def add_model(self, name, model, opt=None):
        assert not name in self.models, "Model {} already added".format(name)

        self.models[name] = {}
        self.models[name]['model'] = model
        self.models[name]['opt'] = opt

    def limit(self):
        return self.max_n

    def add_checkpoints(self, name=None):
        # searching for names
        fns = os.listdir(self.path)
        fns = filter(lambda x: x[-4:] == '.pth', fns)

        names = {}
        for fn in fns:
            sfx = fn.split("_")[-1].rstrip('.pth')
            path = self._get_full_path(fn)
            if not sfx in names:
                names[sfx] = os.path.getmtime(path)
            else:
                names[sfx] = max(names[sfx], os.path.getmtime(path))

        # assembling
        names_and_time = []
        for sfx, time in names.items():
            exists, paths = self.find(sfx)
            if exists:
                names_and_time.append((sfx, time))

        # if there are more checkpoints
        # than we can handle, remove the older ones
        # but do not remove them (for safety)
        if len(names_and_time) > self.max_n:
            names_and_time = sorted(names_and_time, \
                                    key=lambda x: x[1], \
                                    reverse=False)
            new_checkpoints = []
            for key in names_and_time[-self.max_n:]:
                new_checkpoints.append(key[0])

            self.checkpoints = new_checkpoints

    def __len__(self):
        return len(self.checkpoints)

    def _get_full_path(self, filename):
        return os.path.join(self.path, filename)

    def clean(self, n_remove):

        n_remove = min(n_remove, len(self.checkpoints))

        for i in range(n_remove):
            sfx = self.checkpoints[i]

            for name, data in self.models.items():
                for d in ('model', 'opt'):
                    fn = self._filename(d, name, sfx)
                    self._rm(fn)

        removed = self.checkpoints[:n_remove]
        self.checkpoints = self.checkpoints[n_remove:]
        return removed

    def _rm(self, fn):
        path = self._get_full_path(fn)
        if os.path.isfile(path):
            os.remove(path)

    def _filename(self, d, name, suffix):
        return "{}_{}_{}.pth".format(d, name, suffix)

    def load(self, suffix):
        if suffix is None:
            return False

        found, paths = self.find(suffix)
        if not found:
            return False

        # loading
        for name, data in self.models.items():
            for d in ('model', 'opt'):
                if data[d] is not None:
                    data[d].load_state_dict(torch.load(paths[name][d]))

        return True

    def find(self, suffix, force=False):
        paths = {}
        found = True
        for name, data in self.models.items():
            paths[name] = {}
            for d in ('model', 'opt'):
                fn = self._filename(d, name, suffix)
                path = self._get_full_path(fn)
                paths[name][d] = path
                if not os.path.isfile(path):
                    print("File not found: ", path)
                    if d == 'model':
                        found = False

        if found and not suffix in self.checkpoints:
            if len(self.checkpoints) < self.max_n or force:
                self.checkpoints.insert(0, suffix)
                if force:
                    self.max_n = max(self.max_n, len(self.checkpoints))

        return found, paths

    def checkpoint(self, suffix):
        assert not '_' in suffix, "Underscores are not allowed" 

        self.checkpoints.append(suffix)

        for name, data in self.models.items():
            for d in ('model', 'opt'):
                fn = self._filename(d, name, suffix)
                path = self._get_full_path(fn)
                if not os.path.isfile(path) and data[d] is not None:
                    torch.save(data[d].state_dict(), path)

        # removing
        n_remove = max(0, len(self.checkpoints) - self.max_n)
        removed = self.clean(n_remove)

        return removed
