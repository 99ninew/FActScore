import argparse
import os
from pathlib import Path
import sqlite3
from typing import List, Optional


def list_tables(cur) -> List[str]:
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    return [r[0] for r in cur.fetchall()]


def list_columns(cur, table: str) -> List[str]:
    # column info per row: (cid, name, type, notnull, dflt_value, pk)
    cur.execute(f"PRAGMA table_info({table});") 
    return [r[1] for r in cur.fetchall()]


def resolve_db_path(cli_path: Optional[str]) -> Optional[str]:
    candidates: List[Path] = []
    if cli_path:
        candidates.append(Path(cli_path).expanduser())
    # env_path = os.environ.get("FACTSCORE_DB")
    # if env_path:
    #     candidates.append(Path(env_path).expanduser())

    # here = Path(__file__).resolve().parent
    # home = Path.home()
    # likely_rel = [
    #     here / "../.cache/factscore/enwiki-20230401.db",
    #     here / "../../.cache/factscore/enwiki-20230401.db",
    #     here / "enwiki-20230401.db",
    #     here / "../data/enwiki-20230401.db",
    #     home / ".cache/factscore/enwiki-20230401.db",
    # ]
    # candidates.extend(likely_rel)

    for p in candidates:
        if p and p.is_file():
            return str(p)
    return None


def main():
    parser = argparse.ArgumentParser(description="Inspect SQLite KB and optionally query by title")
    parser.add_argument(
        "--db",
        default="../.cache/factscore/enwiki-20230401.db",
        help="Path to SQLite database file",
    )
    parser.add_argument("--table", default=None, help="Table name to inspect (optional)")
    parser.add_argument("--title", default=None, help="Try to fetch content by title (optional)")
    parser.add_argument("--limit", type=int, default=20, help="Number of sample rows to print")
    args = parser.parse_args()

    db_path = resolve_db_path(args.db)
    if not db_path:
        print(
            "Database file not found. Provide --db path or set FACTSCORE_DB. "
            "Tried common locations near this script and ~/.cache/factscore/."
        )
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = list_tables(cursor)
    # print("Tables:", tables)
    if not tables:
        print("No tables found in DB.")
        conn.close()
        return

    table = tables[0] if tables else None
    if not table:
        print("No suitable table found.")
        conn.close()
        return

    # print(f"Using table: {table}")
    # cols = list_columns(cursor, table)
    # print(f"Columns in {table}: {cols}")
    
    # row_count = cursor.execute(f"SELECT COUNT(*) FROM {table};").fetchone()[0]
    # print(f"Total rows in {table}: {row_count}")

    target_entities = []
    with open ("../.cache/factscore/data/labeled/prompt_entities.txt", "r") as f:
        target_entities = f.readlines()

    for title in target_entities:
        title = title.strip()
        cursor.execute(f"SELECT * From {table} WHERE title=?", (title,))
        sample_rows = cursor.fetchone()
        # print(f"Sample {len(sample_rows)} rows from {table} where title={title!r}:")
        sample_text = sample_rows[1] if sample_rows else "N/A"
        print(sample_text)
        # for r in sample_rows:
        #     print(r)
    # cursor.execute(f"SELECT * FROM {table} LIMIT {args.limit};")
    # sample_rows = cursor.fetchall()
    # print(f"Sample {len(sample_rows)} rows from {table}:")
    # for r in sample_rows:
    #     print(r)

    if args.title:
        title_col_candidates = ["title", "page_title", "name", "doc_title"]
        content_col_candidates = ["content", "text", "body", "page_content", "paragraph", "passage"]

        title_col = next((c for c in title_col_candidates if c in cols), None)
        content_col = next((c for c in content_col_candidates if c in cols), None)

        if not title_col:
            print(f"No title-like column found in {table} among {title_col_candidates}.")
        else:
            select_cols = content_col if content_col else "*"
            cursor.execute(
                f"SELECT {select_cols} FROM {table} WHERE {title_col} = ? LIMIT 1;",
                (args.title,),
            )
            row = cursor.fetchone()
            if row is None:
                print(f"No row found where {title_col} == {args.title!r}.")
            else:
                if content_col:
                    print(f"Content from {table}.{content_col} where {title_col} == {args.title!r}:")
                    print(row[0])
                else:
                    print("Row:", row)

    conn.close()


if __name__ == "__main__":
    main()
