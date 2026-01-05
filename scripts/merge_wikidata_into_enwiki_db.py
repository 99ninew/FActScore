#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Reuse the exact constants used by FActScore's DB format
from factscore.retrieval import MAX_LENGTH, SPECIAL_SEPARATOR


@dataclass
class EntityRecord:
    description: Optional[str] = None
    statements: List[Tuple[str, str, Optional[List[dict]]]] = None

    def __post_init__(self):
        if self.statements is None:
            self.statements = []


def iter_wikidata_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def format_qualifiers(qualifiers: Optional[List[dict]]) -> str:
    if not qualifiers:
        return ""
    parts: List[str] = []
    for q in qualifiers:
        prop = q.get("property")
        val = q.get("value")
        if prop is None or val is None:
            continue
        parts.append(f"{prop}: {val}")
    return " (" + "; ".join(parts) + ")" if parts else ""


def build_entity_lines(rec: EntityRecord) -> List[str]:
    lines: List[str] = []
    if rec.description:
        lines.append(f"Description: {rec.description}")
    # deterministic ordering helps reproducibility
    for prop, val, qualifiers in rec.statements:
        q = format_qualifiers(qualifiers)
        lines.append(f"{prop}: {val}{q}")
    return lines


def split_to_sentences(lines: List[str]) -> List[str]:
    # Keep it simple and stable: one fact per "sentence" line.
    # Roberta tokenizer will handle punctuation; DB builder wraps each line in <s>...</s>.
    return [ln.strip() for ln in lines if ln.strip()]


def encode_to_db_text(sentences: List[str], tokenizer) -> str:
    """Match DocDB.build_db() logic: tokenize each sentence, chunk into MAX_LENGTH, decode, join by SPECIAL_SEPARATOR."""
    passages: List[List[int]] = [[]]
    for sent in sentences:
        tokens = tokenizer(sent)["input_ids"]
        max_len = MAX_LENGTH - len(passages[-1])
        if len(tokens) <= max_len:
            passages[-1].extend(tokens)
        else:
            passages[-1].extend(tokens[:max_len])
            offset = max_len
            while offset < len(tokens):
                passages.append(tokens[offset : offset + MAX_LENGTH])
                offset += MAX_LENGTH

    decoded = [
        tokenizer.decode(toks)
        for toks in passages
        if np.sum([t not in [0, 2] for t in toks]) > 0
    ]
    return SPECIAL_SEPARATOR.join(decoded)


def encode_to_db_text_simple(sentences: List[str], max_chars: int = 4000) -> str:
    """Offline-safe fallback: chunk plain text and join by SPECIAL_SEPARATOR.

    This doesn't attempt to reproduce roberta-large token chunking.
    It only preserves the DB contract that passages are separated by SPECIAL_SEPARATOR.
    """
    text = "\n".join([s for s in sentences if s.strip()])
    if not text.strip():
        return ""

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Prefer to break on a newline for readability
        nl = text.rfind("\n", start, end)
        if nl != -1 and nl > start + max_chars // 2:
            end = nl
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end, start + 1)

    return SPECIAL_SEPARATOR.join(chunks)


def title_exists(cur: sqlite3.Cursor, title: str) -> bool:
    cur.execute("SELECT 1 FROM documents WHERE title = ? LIMIT 1;", (title,))
    return cur.fetchone() is not None


def ensure_schema(cur: sqlite3.Cursor):
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='documents';")
    if cur.fetchone() is None:
        raise RuntimeError("Target DB does not have a 'documents' table. Is this an FActScore DB?")


def merge_wikidata_into_db(
    src_db: str,
    out_db: str,
    wikidata_jsonl: str,
    on_conflict: str,
    max_entities: Optional[int],
    commit_every: int,
    mode: str,
) -> Dict[str, int]:
    if not os.path.isfile(src_db):
        raise FileNotFoundError(src_db)
    if not os.path.isfile(wikidata_jsonl):
        raise FileNotFoundError(wikidata_jsonl)

    if os.path.abspath(src_db) != os.path.abspath(out_db):
        os.makedirs(os.path.dirname(os.path.abspath(out_db)), exist_ok=True)
        shutil.copy2(src_db, out_db)

    tokenizer = None
    try:
        from transformers import RobertaTokenizer

        tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    except Exception as e:
        print(
            "Warning: failed to load 'roberta-large' tokenizer; falling back to simple chunking. "
            f"Reason: {type(e).__name__}: {e}"
        )

    conn = sqlite3.connect(out_db)
    cur = conn.cursor()

    ensure_schema(cur)

    stats = {
        "entities_total": 0,
        "inserted": 0,
        "skipped_conflict": 0,
        "replaced": 0,
        "empty_skipped": 0,
    }

    # speed-ups (safe enough for building a derived DB)
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("PRAGMA synchronous=NORMAL;")

    pending: List[Tuple[str, str]] = []

    def flush():
        nonlocal pending
        if not pending:
            return
        cur.executemany("INSERT OR REPLACE INTO documents(title, text) VALUES (?, ?);", pending)
        conn.commit()
        pending = []

    def upsert_entity(ent: str, rec: EntityRecord):
        # sort statements by property then value for stability
        rec.statements.sort(key=lambda x: (x[0], x[1]))

        lines = build_entity_lines(rec)
        sentences = split_to_sentences(lines)
        if not sentences:
            stats["empty_skipped"] += 1
            return

        encoded = encode_to_db_text(sentences, tokenizer) if tokenizer is not None else encode_to_db_text_simple(sentences)
        if not encoded:
            stats["empty_skipped"] += 1
            return

        if on_conflict != "replace" and title_exists(cur, ent):
            if on_conflict == "skip":
                stats["skipped_conflict"] += 1
                return
            if on_conflict == "append":
                cur.execute("SELECT text FROM documents WHERE title = ? LIMIT 1;", (ent,))
                row = cur.fetchone()
                existing_text = row[0] if row else ""
                combined = existing_text + (SPECIAL_SEPARATOR if existing_text and encoded else "") + encoded
                pending.append((ent, combined))
                stats["replaced"] += 1
                return
            raise ValueError(f"Unknown on_conflict mode: {on_conflict}")

        pending.append((ent, encoded))
        if on_conflict == "replace":
            stats["replaced"] += 1
        else:
            stats["inserted"] += 1

    if mode == "gather":
        entities: Dict[str, EntityRecord] = defaultdict(EntityRecord)
        for dp in iter_wikidata_jsonl(wikidata_jsonl):
            ent = dp.get("entity")
            if not ent:
                continue
            if "description" in dp:
                desc = dp.get("description")
                if desc and not entities[ent].description:
                    entities[ent].description = desc
            elif "property" in dp and "value" in dp:
                entities[ent].statements.append((dp["property"], str(dp["value"]), dp.get("qualifiers")))

        entity_names = sorted(entities.keys())
        if max_entities is not None:
            entity_names = entity_names[:max_entities]

        stats["entities_total"] = len(entity_names)
        for idx, ent in enumerate(entity_names, start=1):
            upsert_entity(ent, entities[ent])
            if len(pending) >= commit_every:
                flush()
            if idx % 1000 == 0:
                print(f"Processed {idx}/{len(entity_names)} entities...")
    elif mode == "stream":
        current_ent: Optional[str] = None
        current_rec: Optional[EntityRecord] = None
        n_entities = 0

        for dp in iter_wikidata_jsonl(wikidata_jsonl):
            ent = dp.get("entity")
            if not ent:
                continue

            if current_ent is None:
                current_ent = ent
                current_rec = EntityRecord()

            if ent != current_ent:
                n_entities += 1
                if max_entities is None or n_entities <= max_entities:
                    upsert_entity(current_ent, current_rec)  # type: ignore[arg-type]
                if len(pending) >= commit_every:
                    flush()
                if n_entities % 1000 == 0:
                    print(f"Processed {n_entities} entities...")

                current_ent = ent
                current_rec = EntityRecord()

            # accumulate
            if "description" in dp:
                desc = dp.get("description")
                if desc and not current_rec.description:
                    current_rec.description = desc
            elif "property" in dp and "value" in dp:
                current_rec.statements.append((dp["property"], str(dp["value"]), dp.get("qualifiers")))

        # flush last entity
        if current_ent is not None and current_rec is not None:
            n_entities += 1
            if max_entities is None or n_entities <= max_entities:
                upsert_entity(current_ent, current_rec)

        stats["entities_total"] = min(n_entities, max_entities) if max_entities is not None else n_entities
        if len(pending) >= 1:
            flush()
    else:
        raise ValueError("mode must be 'stream' or 'gather'")

    flush()
    conn.close()
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Merge CKM wikidata.jsonl into an existing FActScore enwiki SQLite DB (documents table)."
    )
    parser.add_argument(
        "--src_db",
        default=".cache/factscore/enwiki-20230401.db",
        help="Path to source enwiki DB (will not be modified).",
    )
    parser.add_argument(
        "--out_db",
        default=".cache/factscore/enwiki-20230401_plus_wikidata.db",
        help="Path to output merged DB.",
    )
    parser.add_argument(
        "--wikidata_jsonl",
        default="../CKM/data/dataset/raw/wikidata.jsonl",
        help="Path to wikidata.jsonl.",
    )
    parser.add_argument(
        "--on_conflict",
        choices=["skip", "replace", "append"],
        default="skip",
        help="What to do if entity title already exists in enwiki DB.",
    )
    parser.add_argument(
        "--max_entities",
        type=int,
        default=None,
        help="For debugging: only merge first N entities (sorted by entity name).",
    )
    parser.add_argument(
        "--commit_every",
        type=int,
        default=2000,
        help="Commit every N inserted rows.",
    )
    parser.add_argument(
        "--mode",
        choices=["stream", "gather"],
        default="stream",
        help="stream: memory-safe, assumes same-entity rows are contiguous; gather: loads all entities then processes (more memory).",
    )
    args = parser.parse_args()

    stats = merge_wikidata_into_db(
        src_db=args.src_db,
        out_db=args.out_db,
        wikidata_jsonl=args.wikidata_jsonl,
        on_conflict=args.on_conflict,
        max_entities=args.max_entities,
        commit_every=args.commit_every,
        mode=args.mode,
    )

    print("Done.")
    for k, v in stats.items():
        print(f"{k}: {v}")
    print(f"Output DB: {args.out_db}")


if __name__ == "__main__":
    main()
