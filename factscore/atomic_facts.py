import json
import numpy as np
import re
import functools
import string
import spacy
import sys
import nltk
import openai
from rank_bm25 import BM25Okapi
import os
import time
import logging
import signal
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

# # Ensure required NLTK punkt models are available. Some environments need 'punkt_tab'.
# nltk.download("punkt", quiet=True)
# try:
#     nltk.download("punkt_tab", quiet=True)
# except Exception:
#     # punkt_tab may not be available in older nltk distributions; we'll fallback at runtime
#     pass


def safe_sent_tokenize(text):
    """Wrap nltk.sent_tokenize and attempt to download punkt_tab on LookupError, then retry."""
    try:
        return sent_tokenize(text)
    except LookupError:
        # Do not attempt network downloads at runtime; fallback to simple regex-based splitter
        return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

from factscore.openai_lm import OpenAIModel



class AtomicFactGenerator(object):
    def __init__(self, key_path, demon_dir, gpt3_cache_file=None, log_file=None):
        self.nlp = spacy.load("en_core_web_sm")
        self.is_bio = True
        self.demon_path = os.path.join(demon_dir, "demons.json" if self.is_bio else "demons_complex.json")

        # 设置日志记录
        if log_file is None:
            log_file = ".cache/factscore/atomic_facts.log"
        
        # 创建日志目录如果不存在
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        self.logger = self._setup_logger(log_file)
        self.log_file = log_file
        
        # API 
        self.api_timeout = 120  # seconds timeout (adjustable)
        self.max_retries = 3    # maximum retry attempts
        self.retry_delay = 5    # retry wait time (seconds)

        # self.openai_lm = OpenAIModel("InstructGPT", cache_file=gpt3_cache_file, key_path=key_path)
        self.openai_lm = OpenAIModel("gpt-oss-120b", cache_file=gpt3_cache_file, key_path=key_path)
        # self.openai_lm = OpenAIModel("deepseek-v3.2", cache_file=gpt3_cache_file, key_path=key_path)

        self.logger.info("=" * 80)
        self.logger.info(f"AtomicFactGenerator initialized at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Model: gpt-oss-120b, Cache file: {gpt3_cache_file}")
        self.logger.info(f"API Timeout: {self.api_timeout}s, Max Retries: {self.max_retries}")
        self.logger.info("=" * 80)

        # get the demons
        with open(self.demon_path, 'r') as f:
            self.demons = json.load(f)

        tokenized_corpus = [doc.split(" ") for doc in self.demons.keys()]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _setup_logger(self, log_file):
        logger = logging.getLogger("AtomicFactGenerator")
        logger.propagate = False  # 禁止传播到根 logger，防止重复输出或绕过 handler 级别限制
        
        if logger.handlers:
            logger.info("Logger already configured, reusing existing setup")
            return logger
        
        logger.setLevel(logging.DEBUG)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.WARNING)
        
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def save_cache(self):
        self.openai_lm.save_cache()

    def run(self, generation, cost_estimate=None):
        """Convert the generation into a set of atomic facts. Return a total words cost if cost_estimate != None."""
        assert isinstance(generation, str), "generation must be a string"
        paragraphs = [para.strip() for para in generation.split("\n") if len(para.strip()) > 0]
        return self.get_atomic_facts_from_paragraph(paragraphs, cost_estimate=cost_estimate)

    def get_atomic_facts_from_paragraph(self, paragraphs, cost_estimate=None):
        sentences = []
        para_breaks = []
        for para_idx, paragraph in enumerate(paragraphs):
            if para_idx > 0 :
                para_breaks.append(len(sentences))

            initials = detect_initials(paragraph)

            curr_sentences = safe_sent_tokenize(paragraph)
            curr_sentences_2 = safe_sent_tokenize(paragraph)

            curr_sentences = fix_sentence_splitter(curr_sentences, initials)
            curr_sentences_2 = fix_sentence_splitter(curr_sentences_2, initials)

            # checking this, just to ensure the crediability of the sentence splitter fixing algorithm
            assert curr_sentences == curr_sentences_2, (paragraph, curr_sentences, curr_sentences_2)

            sentences += curr_sentences

        atoms_or_estimate = self.get_init_atomic_facts_from_sentence([sent for i, sent in enumerate(sentences) if not (not self.is_bio and ( \
                            (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                            (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))))], cost_estimate=cost_estimate)

        if cost_estimate:
            return atoms_or_estimate
        else:
            atoms = atoms_or_estimate

        atomic_facts_pairs = []
        for i, sent in enumerate(sentences):
            if not self.is_bio and ( \
                (i==0 and (sent.startswith("Sure") or sent.startswith("Here are"))) or \
                (i==len(sentences)-1 and (sent.startswith("Please") or sent.startswith("I hope") or sent.startswith("Here are")))):
                atomic_facts_pairs.append((sent, []))
            elif self.is_bio and sent.startswith("This sentence does not contain any facts"):
                atomic_facts_pairs.append((sent, []))
            elif sent.startswith("Sure") or sent.startswith("Please") or (i==0 and sent.startswith("Here are")):
                atomic_facts_pairs.append((sent, []))
            else:
                atomic_facts_pairs.append((sent, atoms[sent]))

        # postprocess_atomic_facts will fix minor issues from InstructGPT
        # it is supposed to handle sentence splitter issue too, but since here
        # we fixed sentence splitter issue already,
        # the new para_breaks should be identical to the original para_breaks
        if self.is_bio:
            atomic_facts_pairs, para_breaks = postprocess_atomic_facts(atomic_facts_pairs, list(para_breaks), self.nlp)

        return atomic_facts_pairs, para_breaks


    def get_init_atomic_facts_from_sentence(self, sentences, cost_estimate=None):
        """Get the initial atomic facts from the sentences. Return a total words cost if cost_estimate != None."""

        is_bio = self.is_bio
        demons = self.demons

        k = 1 if is_bio else 0
        n = 7 if is_bio else 8

        prompts = []
        prompt_to_sent = {}
        atoms = {}
        for sentence in sentences:
            if sentence in atoms:
                continue
            top_machings = best_demos(sentence, self.bm25, list(demons.keys()), k)
            prompt = ""

            for i in range(n):
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(list(demons.keys())[i])
                for fact in demons[list(demons.keys())[i]]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"

            for match in top_machings:
                prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(match)
                for fact in demons[match]:
                    prompt = prompt + "- {}\n".format(fact)
                prompt = prompt + "\n"
            prompt = prompt + "Please breakdown the following sentence into independent facts: {}\n".format(sentence)
            prompts.append(prompt)
            prompt_to_sent[prompt] = sentence

        if cost_estimate:
            total_words_estimate = 0
            for prompt in prompts:
                if cost_estimate == "consider_cache" and (prompt.strip() + "_0") in self.openai_lm.cache_dict:
                    continue
                total_words_estimate += len(prompt.split())
            return total_words_estimate
        else:
            # use tqdm
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Generating atomic facts", unit="sent"), 1):
                sentence = prompt_to_sent[prompt]
                try:
                    self.logger.info(f"\n{'='*80}")
                    self.logger.info(f"Processing sentence {prompt_idx}/{len(prompts)}")
                    self.logger.info(f"Sentence: {sentence}")
                    self.logger.debug(f"Prompt length: {len(prompt.split())} words")
                    
                    # 带重试的 API 调用
                    output = self._call_api_with_retry(prompt, sentence, prompt_idx)
                    atoms[sentence] = text_to_sentences(output)
                    
                    self.logger.info(f"Success - Generated {len(atoms[sentence])} atomic facts:")
                    for fact_idx, fact in enumerate(atoms[sentence], 1):
                        self.logger.info(f"  [{fact_idx}] {fact}")
                    self.logger.debug(f"Full API response: {output[:200]}...")  # 只记录前200字符
                    
                except Exception as e:
                    self.logger.error(f"Error generating atomic facts for sentence: {sentence}")
                    self.logger.error(f"Error type: {type(e).__name__}")
                    self.logger.error(f"Error message: {str(e)}")
                    self.logger.exception("Full traceback:")
                    atoms[sentence] = []  # 在错误情况下设置为空列表

            for key, value in demons.items():
                if key not in atoms:
                    atoms[key] = value

            return atoms

    def _call_api_with_retry(self, prompt, sentence, sentence_idx):
        """带重试机制的 API 调用"""
        for attempt in range(1, self.max_retries + 1):
            try:
                self.logger.info(f"API call attempt {attempt}/{self.max_retries}")
                start_time = time.time()
                
                output, _ = self.openai_lm.generate(prompt)
                
                elapsed_time = time.time() - start_time
                self.logger.info(f"API call succeeded in {elapsed_time:.2f}s")
                return output
                
            except (TimeoutError, ConnectionError) as e:
                # 网络级错误 - 重试
                elapsed_time = time.time() - start_time
                self.logger.warning(f"Attempt {attempt} network error after {elapsed_time:.2f}s: {type(e).__name__}")
                
                if attempt < self.max_retries:
                    self.logger.warning(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed for sentence {sentence_idx}")
                    raise
                    
            except RuntimeError as e:
                # RuntimeError 来自 API 本身（如 response=None）- 重试
                elapsed_time = time.time() - start_time
                error_msg = str(e)
                self.logger.warning(f"Attempt {attempt} API error after {elapsed_time:.2f}s: {error_msg}")
                
                if attempt < self.max_retries:
                    self.logger.warning(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed for sentence {sentence_idx}")
                    raise
                    
            except TypeError as e:
                # TypeError: 'NoneType' object is not subscriptable - API 返回 None
                # 这是可重试的错误，应该重新尝试
                elapsed_time = time.time() - start_time
                error_msg = str(e)
                self.logger.warning(f"Attempt {attempt} got None response after {elapsed_time:.2f}s: {error_msg}")
                
                if attempt < self.max_retries:
                    self.logger.warning(f"API returned None - retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"All {self.max_retries} attempts failed (API returned None) for sentence {sentence_idx}")
                    raise
                    
            except Exception as e:
                # 其他异常 - 分析后决定是否重试或直接失败
                elapsed_time = time.time() - start_time
                error_type = type(e).__name__
                error_msg = str(e)
                
                # 判断是否为可重试的错误
                retryable_keywords = ['timeout', 'connection', 'refused', 'reset', 'broken', 'api error']
                is_retryable = any(keyword in error_msg.lower() for keyword in retryable_keywords)
                
                if is_retryable and attempt < self.max_retries:
                    self.logger.warning(f"Attempt {attempt} retryable error after {elapsed_time:.2f}s: {error_type}: {error_msg}")
                    self.logger.warning(f"Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
                else:
                    self.logger.error(f"Attempt {attempt} non-retryable error after {elapsed_time:.2f}s: {error_type}")
                    self.logger.error(f"Error message: {error_msg}")
                    raise


def best_demos(query, bm25, demons_sents, k):
    tokenized_query = query.split(" ")
    top_machings = bm25.get_top_n(tokenized_query, demons_sents, k)
    return top_machings


# transform InstructGPT output into sentences
def text_to_sentences(text):
    sentences = text.split("- ")[1:]
    sentences = [sent.strip()[:-1] if sent.strip()[-1] == '\n' else sent.strip() for sent in sentences]
    if len(sentences) > 0: 
        if sentences[-1][-1] != '.':
            sentences[-1] = sentences[-1] + '.' 
    else:
        sentences = []
    return sentences


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTHS = [m.lower() for m in MONTHS]

def is_num(text):
    try:
        text = int(text)
        return True
    except Exception:
        return False

def is_date(text):
    text = normalize_answer(text)
    for token in text.split(" "):
        if (not is_num(token)) and token not in MONTHS:
            return False
    return True

def extract_numeric_values(text):
    pattern = r'\b\d+\b'  # regular expression pattern for integers
    numeric_values = re.findall(pattern, text)  # find all numeric values in the text
    return set([value for value in numeric_values])  # convert the values to float and return as a list


def detect_entities(text, nlp):
    doc = nlp(text)
    entities = set()

    def _add_to_entities(text):
        if "-" in text:
            for _text in text.split("-"):
                entities.add(_text.strip())
        else:
            entities.add(text)


    for ent in doc.ents:
        # spacy often has errors with other types of entities
        if ent.label_ in ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]:

            if is_date(ent.text):
                _add_to_entities(ent.text)
            else:
                for token in ent.text.split():
                    if is_date(token):
                        _add_to_entities(token)
        
    for new_ent in extract_numeric_values(text):
        if not np.any([new_ent in ent for ent in entities]):
            entities.add(new_ent)

    return entities

def postprocess_atomic_facts(_atomic_facts, para_breaks, nlp):

    verbs = ["born.", " appointed.", " characterized.", " described.", " known.", " member.", " advocate.", "served.", "elected."]
    permitted_verbs = ["founding member."]

    atomic_facts = []
    new_atomic_facts = []
    new_para_breaks = []

    for i, (sent, facts) in enumerate(_atomic_facts):
        sent = sent.strip()
        if len(sent.split())==1 and i not in para_breaks and i > 0:
            assert i not in para_breaks
            atomic_facts[-1][0] += " " + sent
            atomic_facts[-1][1] += facts
        else:
            if i in para_breaks:
                new_para_breaks.append(len(atomic_facts))
            atomic_facts.append([sent, facts])

    for i, (sent, facts) in enumerate(atomic_facts):
        entities = detect_entities(sent, nlp)
        covered_entities = set()
        # print (entities)
        new_facts = []
        for i, fact in enumerate(facts):
            if any([fact.endswith(verb) for verb in verbs]) and not any([fact.endswith(verb) for verb in permitted_verbs]):
                if any([fact[:-1] in other_fact for j, other_fact in enumerate(facts) if j != i]):
                    continue
            sent_entities = detect_entities(fact, nlp)
            covered_entities |= set([e for e in sent_entities if e in entities])
            new_entities = sent_entities - entities
            if len(new_entities) > 0:
                do_pass = False
                for new_ent in new_entities:
                    pre_ent = None
                    for ent in entities:
                        if ent.startswith(new_ent):
                            pre_ent = ent
                            break
                    if pre_ent is None:
                        do_pass = True
                        break
                    fact = fact.replace(new_ent, pre_ent)
                    covered_entities.add(pre_ent)
                if do_pass:
                    continue
            if fact in new_facts:
                continue
            new_facts.append(fact)
        try:
            assert entities==covered_entities
        except Exception:
            new_facts = facts # there is a bug in spacy entity linker, so just go with the previous facts

        new_atomic_facts.append((sent, new_facts))

    return new_atomic_facts, new_para_breaks

def is_integer(s):
    try:
        s = int(s)
        return True
    except Exception:
        return False

def detect_initials(text):
    pattern = r"[A-Z]\. ?[A-Z]\."
    match = re.findall(pattern, text)
    return [m for m in match]

def fix_sentence_splitter(curr_sentences, initials):
    for initial in initials:
        if not np.any([initial in sent for sent in curr_sentences]):
            alpha1, alpha2 = [t.strip() for t in initial.split(".") if len(t.strip())>0]
            for i, (sent1, sent2) in enumerate(zip(curr_sentences, curr_sentences[1:])):
                if sent1.endswith(alpha1 + ".") and sent2.startswith(alpha2 + "."):
                    # merge sentence i and i+1
                    curr_sentences = curr_sentences[:i] + [curr_sentences[i] + " " + curr_sentences[i+1]] + curr_sentences[i+2:]
                    break
    sentences = []
    combine_with_previous = None
    for sent_idx, sent in enumerate(curr_sentences):
        if len(sent.split())<=1 and sent_idx==0:
            assert not combine_with_previous
            combine_with_previous = True
            sentences.append(sent)
        elif len(sent.split())<=1:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combined_with_previous = False
        elif sent[0].isalpha() and not sent[0].isupper() and sent_idx > 0:
            assert sent_idx > 0, curr_sentences
            sentences[-1] += " " + sent
            combine_with_previous = False
        elif combine_with_previous:
            assert sent_idx > 0
            sentences[-1] += " " + sent
            combine_with_previous = False
        else:
            assert not combine_with_previous
            sentences.append(sent)
    return sentences


def main():
    generator = AtomicFactGenerator(".cache/factscore/api_key.txt", ".cache/factscore/demos", gpt3_cache_file=os.path.join(".cache/factscore", "deepseek_v3dot2.pkl"))
    atomic_facts, para_breaks = generator.run("Thierry Henry (born 17 August 1977) is a French professional football coach, pundit, and former player. He is considered one of the greatest strikers of all time, and one the greatest players of the Premier League history. He has been named Arsenal F.C's greatest ever player.\n\nHenry made his professional debut with Monaco in 1994 before signing for defending Serie A champions Juventus. However, limited playing time, coupled with disagreements with the club's hierarchy, led to him signing for Premier League club Arsenal for £11 million in 1999.")

    print(atomic_facts)
    print(para_breaks)

if __name__ == "__main__":
    main()