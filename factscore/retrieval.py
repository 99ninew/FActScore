import json
import time
import os

import sqlite3
import numpy as np
import pickle as pkl
import json
import tempfile
import shutil

from rank_bm25 import BM25Okapi

SPECIAL_SEPARATOR = "####SPECIAL####SEPARATOR####"
MAX_LENGTH = 256

class DocDB(object):
    """Sqlite backed document storage.

    Implements get_doc_text(doc_id).
    """

    def __init__(self, db_path=None, data_path=None):
        self.db_path = db_path
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)

        cursor = self.connection.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        
        if len(cursor.fetchall())==0:
            assert data_path is not None, f"{self.db_path} is empty. Specify `data_path` in order to create a DB."
            print (f"{self.db_path} is empty. start building DB from {data_path}...")
            self.build_db(self.db_path, data_path)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def path(self):
        """Return the path to the file that backs this database."""
        return self.path

    def close(self):
        """Close the connection to the database."""
        self.connection.close()

    def build_db(self, db_path, data_path):
        from transformers import RobertaTokenizer
        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
        
        titles = set()
        output_lines = []
        tot = 0
        start_time = time.time()
        c = self.connection.cursor()
        c.execute("CREATE TABLE documents (title PRIMARY KEY, text);")

        with open(data_path, "r") as f:
            for line in f:
                dp = json.loads(line)
                title = dp["title"]
                text = dp["text"]
                if title in titles:
                    continue
                titles.add(title)
                if type(text)==str:
                    text = [text]
                passages = [[]]
                for sent_idx, sent in enumerate(text):
                    assert len(sent.strip())>0
                    tokens = tokenizer(sent)["input_ids"]
                    max_length = MAX_LENGTH - len(passages[-1])
                    if len(tokens) <= max_length:
                        passages[-1].extend(tokens)
                    else:
                        passages[-1].extend(tokens[:max_length])
                        offset = max_length
                        while offset < len(tokens):
                            passages.append(tokens[offset:offset+MAX_LENGTH])
                            offset += MAX_LENGTH
                
                psgs = [tokenizer.decode(tokens) for tokens in passages if np.sum([t not in [0, 2] for t in tokens])>0]
                text = SPECIAL_SEPARATOR.join(psgs)
                output_lines.append((title, text))
                tot += 1

                if len(output_lines) == 1000000:
                    c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
                    output_lines = []
                    print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        if len(output_lines) > 0:
            c.executemany("INSERT INTO documents VALUES (?,?)", output_lines)
            print ("Finish saving %dM documents (%dmin)" % (tot / 1000000, (time.time()-start_time)/60))

        self.connection.commit()
        self.connection.close()

    def get_text_from_title(self, title):
        """Fetch the raw text of the doc for 'doc_id'."""
        cursor = self.connection.cursor()
        cursor.execute("SELECT text FROM documents WHERE title = ?", (title,))
        results = cursor.fetchall()
        results = [r for r in results]
        cursor.close()
        # assert results is not None and len(results)==1, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        if results is None or len(results) != 1:
            print(f"Topic '{title}' not found in DB, skipping.")
            return None
        results = [{"title": title, "text": para} for para in results[0][0].split(SPECIAL_SEPARATOR)]
        assert len(results)>0, f"`topic` in your data ({title}) is likely to be not a valid title in the DB."
        return results

class Retrieval(object):

    def __init__(self, db, cache_path, embed_cache_path,
                 retrieval_type="gtr-t5-large", batch_size=None):
        self.db = db
        self.cache_path = cache_path
        self.embed_cache_path = embed_cache_path
        self.retrieval_type = retrieval_type
        self.batch_size = batch_size
        assert retrieval_type=="bm25" or retrieval_type.startswith("gtr-")
        
        self.encoder = None
        self.load_cache()
        self.add_n = 0
        self.add_n_embed = 0

    def load_encoder(self):
        from sentence_transformers import SentenceTransformer
        encoder = SentenceTransformer("sentence-transformers/" + self.retrieval_type)
        # Allow choosing a specific device (e.g., keep retrieval on GPU0)
        # Examples: FACTSCORE_RETRIEVAL_DEVICE=cuda:0 / cuda:1 / cpu
        retrieval_device = os.environ.get("FACTSCORE_RETRIEVAL_DEVICE")
        if retrieval_device is None:
            retrieval_device = "cuda:0" if os.environ.get("CUDA_VISIBLE_DEVICES") is None else "cuda"
        encoder = encoder.to(retrieval_device)
        encoder = encoder.eval()
        self.encoder = encoder
        assert self.batch_size is not None
    
    def load_cache(self):
        self.cache = {}
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                # Cache file may be partially written (e.g., process interrupted), back it up and continue.
                backup_path = self.cache_path + ".corrupt"
                try:
                    shutil.copy2(self.cache_path, backup_path)
                except Exception:
                    pass
                print(f"Warning: failed to load cache JSON {self.cache_path!r} ({type(e).__name__}: {e}). Resetting cache; backed up to {backup_path!r} if possible.")
                self.cache = {}

        self.embed_cache = {}
        if os.path.exists(self.embed_cache_path):
            try:
                with open(self.embed_cache_path, "rb") as f:
                    self.embed_cache = pkl.load(f)
            except Exception as e:
                backup_path = self.embed_cache_path + ".corrupt"
                try:
                    shutil.copy2(self.embed_cache_path, backup_path)
                except Exception:
                    pass
                print(f"Warning: failed to load embed cache {self.embed_cache_path!r} ({type(e).__name__}: {e}). Resetting embed cache; backed up to {backup_path!r} if possible.")
                self.embed_cache = {}
    
    def save_cache(self):
        if self.add_n > 0:
            if os.path.exists(self.cache_path):
                try:
                    with open(self.cache_path, "r", encoding="utf-8") as f:
                        new_cache = json.load(f)
                    self.cache.update(new_cache)
                except Exception as e:
                    backup_path = self.cache_path + ".corrupt"
                    try:
                        shutil.copy2(self.cache_path, backup_path)
                    except Exception:
                        pass
                    print(
                        f"Warning: failed to merge existing cache JSON {self.cache_path!r} "
                        f"({type(e).__name__}: {e}). Ignoring old cache; backed up to {backup_path!r} if possible."
                    )

            # Atomic write to avoid corrupt JSON on interruption
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(self.cache_path) + ".", dir=os.path.dirname(self.cache_path) or None)
            try:
                with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f)
                os.replace(tmp_path, self.cache_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
        
        if self.add_n_embed > 0:
            if os.path.exists(self.embed_cache_path):
                try:
                    with open(self.embed_cache_path, "rb") as f:
                        new_cache = pkl.load(f)
                    self.embed_cache.update(new_cache)
                except Exception as e:
                    backup_path = self.embed_cache_path + ".corrupt"
                    try:
                        shutil.copy2(self.embed_cache_path, backup_path)
                    except Exception:
                        pass
                    print(
                        f"Warning: failed to merge existing embed cache {self.embed_cache_path!r} "
                        f"({type(e).__name__}: {e}). Ignoring old embed cache; backed up to {backup_path!r} if possible."
                    )

            # Atomic write for pickle as well
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=os.path.basename(self.embed_cache_path) + ".", dir=os.path.dirname(self.embed_cache_path) or None)
            try:
                with os.fdopen(tmp_fd, "wb") as f:
                    pkl.dump(self.embed_cache, f)
                os.replace(tmp_path, self.embed_cache_path)
            finally:
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass

    def get_bm25_passages(self, topic, query, passages, k):
        if topic in self.embed_cache:
            bm25 = self.embed_cache[topic]
        else:
            bm25 = BM25Okapi([psg["text"].replace("<s>", "").replace("</s>", "").split() for psg in passages])
            self.embed_cache[topic] = bm25
            self.add_n_embed += 1
        scores = bm25.get_scores(query.split())
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_gtr_passages(self, topic, retrieval_query, passages, k):
        if self.encoder is None:
            self.load_encoder()
        if topic in self.embed_cache:
            passage_vectors = self.embed_cache[topic]
        else:
            inputs = [psg["title"] + " " + psg["text"].replace("<s>", "").replace("</s>", "") for psg in passages]
            passage_vectors = self.encoder.encode(inputs, batch_size=self.batch_size, device=self.encoder.device)
            self.embed_cache[topic] = passage_vectors
            self.add_n_embed += 1
        query_vectors = self.encoder.encode([retrieval_query], 
                                            batch_size=self.batch_size,
                                            device=self.encoder.device)[0]
        scores = np.inner(query_vectors, passage_vectors)
        indices = np.argsort(-scores)[:k]
        return [passages[i] for i in indices]

    def get_passages(self, topic, question, k):
        retrieval_query = topic + " " + question.strip()
        cache_key = topic + "#" + retrieval_query
        
        if cache_key not in self.cache:
            passages = self.db.get_text_from_title(topic)
            # If the topic is not found in the DB, get_text_from_title returns None.
            # Guard against passing None into the retrieval methods which expect an iterable.
            if passages is None:
                print(f"Topic '{topic}' not found in DB, skipping retrieval.")
                # cache an empty list so subsequent calls are fast
                self.cache[cache_key] = []
                self.add_n += 1
                return self.cache[cache_key]

            if self.retrieval_type=="bm25":
                self.cache[cache_key] = self.get_bm25_passages(topic, retrieval_query, passages, k)
            else:
                self.cache[cache_key] = self.get_gtr_passages(topic, retrieval_query, passages, k)
            # if passages is an empty list, allow empty cache result; otherwise length should match
            if len(passages) > 0:
                assert len(self.cache[cache_key]) in [k, len(passages)]
            self.add_n += 1
        
            
        return self.cache[cache_key]

        
        


