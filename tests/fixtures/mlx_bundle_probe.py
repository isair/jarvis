import mlx.core as mx
import mlx_whisper
import numpy as np
from mlx_whisper.tokenizer import get_tokenizer
from mlx_whisper.timing import dtw_cpu
value=mx.sum(mx.ones(8))
mx.eval(value)
assert float(value)==8
text='Jarvis speech packaging'
tokenizer=get_tokenizer(multilingual=True,language='en',task='transcribe')
assert tokenizer.decode(tokenizer.encode(text))==text
assert len(dtw_cpu(np.zeros((2,4),dtype=np.float32)))==2
print('✅ FROZEN_MLX_METAL_PASSED',flush=True)

# Optional offline speech fixture: "Jarvis, please tell me the current time."
import sys
if len(sys.argv)==3:
    result=mlx_whisper.transcribe(np.load(sys.argv[2]),path_or_hf_repo=sys.argv[1],language='en')
    transcript=result['text'].strip()
    assert 'tell me the current time' in transcript.lower(), transcript
    print(f'✅ FROZEN_MLX_TRANSCRIPTION_PASSED: {transcript}',flush=True)
