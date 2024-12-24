python tools/preprocess_data.py \
       --input "/datasets/MegatronTest/Alpaca ZH Demo.jsonl" \
       --output-prefix /datasets/MegatronTest/EMU3_TestDATA \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model /data/models/Emu3-Stage1 \
       --json-keys text_prefix text \
       --workers 4 \
       --partitions 2