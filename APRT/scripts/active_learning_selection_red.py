import sys
import json

input = open(sys.argv[1], 'r')
output = open(sys.argv[2], 'w')
safety_threshold = float(sys.argv[3])
helpfulness_threshold = float(sys.argv[4])
max_count = int(sys.argv[5])

count = 0
last_as = 0
while 1:
    line = input.readline()
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    json_line = json.loads(line)
    as_count = int(json_line["as_count"])
    last_as = as_count
    helpful_score = json_line["helpful_score"]
    safety_score = json_line["safety_score"]
    conversations = json_line["conversations"]
    if helpful_score < helpfulness_threshold or safety_score > safety_threshold:
        continue
    if count >= max_count:
        break
    instruction = conversations[0]["content"]
    output_str = conversations[1]["content"]
    output.write(json.dumps({"instruction": instruction, "output": output_str}, ensure_ascii=False) + "\n")
    output.flush()
    count += 1

print("Red LLM active learning data selection, last_as_count:", last_as)

input.close()
