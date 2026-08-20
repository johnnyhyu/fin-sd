import json

INPUT_FILE_PATH = "robustness/Robustness_evaluation_220.gpt-4.1.llm6output.postprocessed_with_ratings.json"  

def load_jsonl_or_json(file_path):
    with open(file_path, encoding="utf-8") as f:
        if file_path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                lines = [line for line in f if line.strip()]
                try:
                    return [json.loads(line) for line in lines]
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to parse file {file_path}. The file is neither valid JSON nor JSONL.\nError: {e}"
                    )

def calculate_overall_average_scores(items):
    answerable_scores = []
    not_answerable_scores = []
    
    for item in items:
        answerable_avg = item.get("answerable_avg_score")
        not_answerable_avg = item.get("not_answerable_avg_score")

        if answerable_avg is not None:
            try:
                answerable_scores.append(float(answerable_avg))
            except (ValueError, TypeError):
                print(f"Warning: Could not convert answerable_avg_score '{answerable_avg}' to float")
        
        if not_answerable_avg is not None:
            try:
                not_answerable_scores.append(float(not_answerable_avg))
            except (ValueError, TypeError):
                print(f"Warning: Could not convert not_answerable_avg_score '{not_answerable_avg}' to float")
    
    overall_answerable_avg = sum(answerable_scores) / len(answerable_scores) if answerable_scores else None
    overall_not_answerable_avg = sum(not_answerable_scores) / len(not_answerable_scores) if not_answerable_scores else None
    
    return {
        "overall_answerable_avg_score": format(overall_answerable_avg, '.4f') if overall_answerable_avg is not None else None,
        "overall_not_answerable_avg_score": format(overall_not_answerable_avg, '.4f') if overall_not_answerable_avg is not None else None,
        "answerable_count": len(answerable_scores),
        "not_answerable_count": len(not_answerable_scores)
    }

def append_overall_scores_to_file(file_path, overall_scores):
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error: The file {file_path} is not a valid JSON file.")
            return False
    
    if isinstance(data, list):
        data.append({
            "is_overall_summary": True,
            **overall_scores
        })
    elif isinstance(data, dict):
        data.update({
            "overall_summary": overall_scores
        })
    else:
        print(f"Error: Unexpected data structure in {file_path}")
        return False
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    return True

def main():
    print(f"Loading data from: {INPUT_FILE_PATH}")
    items = load_jsonl_or_json(INPUT_FILE_PATH)
    print(f"Loaded {len(items)} items")
    
    results = calculate_overall_average_scores(items)
    
    print("\nOVERALL AVERAGE SCORES:")
    print(f"Answerable items: {results['overall_answerable_avg_score']} (based on {results['answerable_count']} items)")
    print(f"Not answerable items: {results['overall_not_answerable_avg_score']} (based on {results['not_answerable_count']} items)")
    
    if append_overall_scores_to_file(INPUT_FILE_PATH, results):
        print(f"\nOverall scores appended to: {INPUT_FILE_PATH}")
    else:
        print(f"\nFailed to append scores to: {INPUT_FILE_PATH}")

if __name__ == "__main__":
    main()