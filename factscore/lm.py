import pickle
import os
import time
import threading
import tempfile

class LM(object):

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0
        self.lock = threading.RLock()
        self._model_init_lock = threading.Lock()

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        with self.lock:
            if cache_key in self.cache_dict:
                return self.cache_dict[cache_key]

        if self.model is None:
            # 多线程下防止重复加载模型（否则会导致显存暴涨/权重处于 meta 状态等问题）
            with self._model_init_lock:
                if self.model is None:
                    self.load_model()

        if prompt.endswith(" True or False?\nAnswer:"):
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        with self.lock:
            self.cache_dict[cache_key] = generated
            self.add_n += 1
        return generated

    def save_cache(self):
        # Take a snapshot under lock so other threads can keep running.
        with self.lock:
            if self.add_n == 0:
                return

        # Merge in any cache written by other processes.
        try:
            latest = self.load_cache(allow_retry=False)
        except Exception:
            latest = {}

        with self.lock:
            if latest:
                self.cache_dict.update(latest)
            snapshot = dict(self.cache_dict)

        # Atomic write so readers never see a half-written pickle.
        tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(self.cache_file) + ".", dir=os.path.dirname(self.cache_file) or None)
        try:
            with os.fdopen(tmp_fd, "wb") as f:
                pickle.dump(snapshot, f)
            os.replace(tmp_path, self.cache_file)
        finally:
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        with self.lock:
            self.add_n = 0

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache



