import os

def main():
    cur_dir = os.getcwd()
    org_file = os.path.join(cur_dir, 'enwiki_text_only.txt')
    entity_file = os.path.join(cur_dir, '.cache/factscore/data/labeled/prompt_entities.txt')
    
    filter_dir = os.path.join(cur_dir, 'enwiki_filtered')
    os.makedirs(filter_dir, exist_ok=True)
    
    if not os.path.exists(org_file):
        print(f"Original file {org_file} does not exist.")
        return
    
    cnt = 0
    with open(org_file, 'r', encoding='utf-8') as f, open(entity_file, 'r', encoding='utf-8') as ef:
        for line in f:
            line = line.strip()
            if not line:
                continue
            filtered_line = line.replace('####SPECIAL####SEPARATOR####', '')
            
            title = ef.readline().strip()
            new_file = os.path.join(filter_dir, f'{title}.txt')
            with open(new_file, 'a', encoding='utf-8') as nf:
                nf.write(filtered_line)
                # print("write to ", new_file)
                cnt += 1

    print("Filtering completed. Total lines processed:", cnt)
    
if __name__ == "__main__":
    main()