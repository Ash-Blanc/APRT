#!/bin/bash

set -e

# Training-Free Automated Progressive Red Teaming (TF-APRT) Runner
# This script provides easy ways to run TF-APRT with different models and configurations

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Training-Free APRT Runner${NC}"
echo -e "${BLUE}========================================${NC}"

# Default values
MODEL_TYPE=""
MODEL_PATH=""
INPUT_FILE=""
OUTPUT_DIR="tf_aprt_results"
CONFIG_FILE="configs/tf_aprt_config.json"
API_KEY=""
NUM_QUERIES=""

# Function to print usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --model-type TYPE     Model type: vllm, openai, anthropic"
    echo "  -m, --model-path PATH     Path to model or model name"
    echo "  -i, --input-file FILE     Input file with queries"
    echo "  -o, --output-dir DIR      Output directory (default: tf_aprt_results)"
    echo "  -c, --config-file FILE    Configuration file (default: configs/tf_aprt_config.json)"
    echo "  -k, --api-key KEY         API key for cloud models"
    echo "  -n, --num-queries NUM     Limit number of queries to process"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Local VLLM model"
    echo "  $0 -t vllm -m /path/to/llama-model -i advbench_queries.txt"
    echo ""
    echo "  # OpenAI GPT-4"
    echo "  $0 -t openai -m gpt-4 -i queries.json -k your_api_key"
    echo ""
    echo "  # Anthropic Claude"
    echo "  $0 -t anthropic -m claude-3-sonnet-20240229 -i queries.txt -k your_api_key"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--model-type)
            MODEL_TYPE="$2"
            shift 2
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -i|--input-file)
            INPUT_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -c|--config-file)
            CONFIG_FILE="$2"
            shift 2
            ;;
        -k|--api-key)
            API_KEY="$2"
            shift 2
            ;;
        -n|--num-queries)
            NUM_QUERIES="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$MODEL_TYPE" ]; then
    echo -e "${RED}Error: Model type is required${NC}"
    usage
fi

if [ -z "$MODEL_PATH" ]; then
    echo -e "${RED}Error: Model path is required${NC}"
    usage
fi

if [ -z "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file is required${NC}"
    usage
fi

# Check if input file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo -e "${RED}Error: Input file '$INPUT_FILE' not found${NC}"
    exit 1
fi

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${YELLOW}Warning: Config file '$CONFIG_FILE' not found, will use defaults${NC}"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo -e "${GREEN}Configuration:${NC}"
echo "  Model Type: $MODEL_TYPE"
echo "  Model Path: $MODEL_PATH"
echo "  Input File: $INPUT_FILE"
echo "  Output Dir: $OUTPUT_DIR"
echo "  Config File: $CONFIG_FILE"
if [ -n "$API_KEY" ]; then
    echo "  API Key: [REDACTED]"
fi
if [ -n "$NUM_QUERIES" ]; then
    echo "  Max Queries: $NUM_QUERIES"
fi
echo ""

# Prepare optional arguments
OPTIONAL_ARGS=""
if [ -n "$API_KEY" ]; then
    OPTIONAL_ARGS="$OPTIONAL_ARGS --api_key $API_KEY"
fi

# Limit queries if specified
TEMP_INPUT_FILE="$INPUT_FILE"
if [ -n "$NUM_QUERIES" ]; then
    echo -e "${YELLOW}Limiting to first $NUM_QUERIES queries...${NC}"
    TEMP_INPUT_FILE="${OUTPUT_DIR}/limited_queries.txt"
    
    if [[ "$INPUT_FILE" == *.json ]]; then
        # Handle JSON files
        python3 -c "
import json
import sys
with open('$INPUT_FILE', 'r') as f:
    data = json.load(f)
    queries = data if isinstance(data, list) else data.get('queries', [])
    limited = queries[:$NUM_QUERIES]
    with open('$TEMP_INPUT_FILE', 'w') as out:
        json.dump(limited, out, indent=2)
"
    else
        # Handle text files
        head -n "$NUM_QUERIES" "$INPUT_FILE" > "$TEMP_INPUT_FILE"
    fi
fi

echo -e "${BLUE}Running Training-Free APRT...${NC}"

# Run the TF-APRT integration script
python3 scripts/tf_aprt_integration.py \
    --model_type "$MODEL_TYPE" \
    --model_path "$MODEL_PATH" \
    --input_file "$TEMP_INPUT_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --config_file "$CONFIG_FILE" \
    $OPTIONAL_ARGS

# Check if the run was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ TF-APRT completed successfully!${NC}"
    echo -e "${GREEN}Results saved to: $OUTPUT_DIR${NC}"
    
    # Show quick stats if comprehensive results exist
    if [ -f "$OUTPUT_DIR/comprehensive_results.json" ]; then
        echo ""
        echo -e "${BLUE}Quick Statistics:${NC}"
        python3 -c "
import json
try:
    with open('$OUTPUT_DIR/comprehensive_results.json', 'r') as f:
        data = json.load(f)
    tf_results = data['tf_aprt_results']
    total = tf_results['total_queries']
    successful = tf_results['successful_attacks']
    rate = successful / total * 100 if total > 0 else 0
    print(f'  Total Queries: {total}')
    print(f'  Successful Attacks: {successful}')
    print(f'  Success Rate: {rate:.1f}%')
except Exception as e:
    print(f'  Error reading results: {e}')
"
    fi
    
    # Display available files
    echo ""
    echo -e "${BLUE}Generated Files:${NC}"
    if [ -f "$OUTPUT_DIR/tf_aprt_report.txt" ]; then
        echo "  📊 Report: $OUTPUT_DIR/tf_aprt_report.txt"
    fi
    if [ -f "$OUTPUT_DIR/comprehensive_results.json" ]; then
        echo "  📋 Full Results: $OUTPUT_DIR/comprehensive_results.json"
    fi
    if [ -f "$OUTPUT_DIR/aprt_format_results.jsonl" ]; then
        echo "  🔄 APRT Compatible: $OUTPUT_DIR/aprt_format_results.jsonl"
    fi
    
else
    echo ""
    echo -e "${RED}✗ TF-APRT failed. Check the error messages above.${NC}"
    exit 1
fi

# Clean up temporary files
if [ "$TEMP_INPUT_FILE" != "$INPUT_FILE" ] && [ -f "$TEMP_INPUT_FILE" ]; then
    rm "$TEMP_INPUT_FILE"
fi

echo ""
echo -e "${GREEN}Done!${NC}"