# Training Emu3-Gen for video generation in Megatron-LM
## Weight-conversion
```python
python tools/checkpoint/convert.py --model-type GPT --loader emu3 --saver mcore --load-dir /data/models/Emu3-Gen --save-dir /data/models/Emu3-Gen-Megatron --model-size emu3-gen --make-vocab-size-divisible-by 2 --saver-transformer-impl transformer_engine
```
