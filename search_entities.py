import argparse
import os
import sys
from typing import Set, List
import json

def load_entities(entities_file: str) -> Set[str]:
    entities = set()
    try:
        with open(entities_file, 'r', encoding='utf-8') as f:
            for line in f:
                entity = line.strip()
                if entity: 
                    entities.add(entity)
        return entities
    except FileNotFoundError:
        print(f"error: Cannot found {entities_file}")
        sys.exit(1)
    except Exception as e:
        print(f"error - {e}")
        sys.exit(1)
        
def search_entities_in_json(json_file: str, entities: Set[str], output_file: str) -> None:
    found_entities = set()
    found_count = 0
           
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    matched_items = []
    for cid, cdata in data.get("concepts", {}).items():
        item_str = json.dumps(cdata, ensure_ascii=False)
        for entity in entities:
            if entity in item_str:
                matched_items.append(cdata)
                found_entities.add(entity)
                found_count += 1
                print(f"find '{entity}' in concepts")
                print(f"find '{cid}' corresponding to concept")
                break 
    for eid, edata in data.get("entities", {}).items():
        item_str = json.dumps(edata, ensure_ascii=False)
        for entity in entities:
            if entity in item_str:
                matched_items.append(edata)
                found_entities.add(entity)
                found_count += 1
                print(f"find '{entity}' in entities")
                print(f"find '{eid}' corresponding to entity")
                break
    
    print(f"search completed, found {len(matched_items)} matching items")
    print(f"number of matched entities: {len(found_entities)}")
    print(f"total number of found entities: {found_count}")

def search_and_copy(target_file: str, entities: Set[str], output_file: str) -> None:
    matched_lines = []
    found_entities = set()
    
    try:        
        with open(target_file, 'r', encoding='utf-8') as f:
            line_number = 0
            for line in f:
                line_number += 1
                line_content = line.rstrip('\n\r')  
                
                for entity in entities:
                    if entity in line_content:
                        matched_lines.append(line_content)
                        found_entities.add(entity)
                        print(f"line {line_number} finds '{entity}'")
                        break 
                
                if line_number % 1000 == 0:
                    print(f"processed {line_number} lines...")

        print(f"search completed, found {len(matched_lines)} matching lines")
        print(f"number of matched entities: {len(found_entities)}")

    except FileNotFoundError:
        print(f"error: Cannot found {target_file}")
        sys.exit(1)
    except Exception as e:
        print(f"error - {e}")
        sys.exit(1)
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in matched_lines:
                f.write(line + '\n')
            f.write("\n# Unmatched entities:\n")
            for entity in entities:
                if entity not in found_entities:
                    f.write(f"# {entity}\n")

        print(f"Results are saved to {output_file}")

        print(f"\nStatistics:")
        print(f"- Total reference entities: {len(entities)}")
        print(f"- Found entities: {len(found_entities)}")
        print(f"- Matched lines: {len(matched_lines)}")
        print(f"- Match rate: {len(found_entities)/len(entities)*100:.2f}%")
        
    except Exception as e:
        print(f"error - {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""examples:
  python search_entities.py entities_list.txt target_file.txt output.txt
  python search_entities.py --entities entities.txt --target data.txt --output results.txt
        """
    )
    
    parser.add_argument('entities_file', nargs='?')
    parser.add_argument('target_file', nargs='?')
    parser.add_argument('output_file', nargs='?')
    
    parser.add_argument('--entities', '-e')
    parser.add_argument('--target', '-t')
    parser.add_argument('--output', '-o')

    args = parser.parse_args()
    
    entities_file = args.entities_file or args.entities
    target_file = args.target_file or args.target
    output_file = args.output_file or args.output
    
    if not all([entities_file, target_file, output_file]):
        print("error: Need to provide all three parameters")
        parser.print_help()
        sys.exit(1)
    
    if not os.path.exists(entities_file):
        print(f"error: Cannot found {entities_file}")
        sys.exit(1)
    
    if not os.path.exists(target_file):
        print(f"error: Cannot found {target_file}")
        sys.exit(1)
    
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
   
    entities = load_entities(entities_file)
    # entities = {"Q4776574", "Q30006721", "Q63019116","Q20716310", "Q6003686", "Q78721193", "Q56605091", "Q16125484", "Q4013563"}
    # search_and_copy(target_file, entities, output_file)
    search_entities_in_json(target_file, entities, output_file)
    
    print("All tasks completed.")

if __name__ == "__main__":
    main()