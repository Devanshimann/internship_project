from pathlib import Path
import io
import tokenize
from tokenize import untokenize

path = Path('c:/Users/itsde/OneDrive/Desktop/internship linux world/project.py')
source = path.read_text(encoding='utf-8')
reader = io.BytesIO(source.encode('utf-8')).readline
output_tokens = []

for tok in tokenize.tokenize(reader):
    if tok.type in (tokenize.COMMENT, tokenize.ENCODING, tokenize.ENDMARKER):
        continue
    output_tokens.append((tok.type, tok.string))

cleaned = untokenize(output_tokens)
if isinstance(cleaned, bytes):
    cleaned = cleaned.decode('utf-8')

path.write_text(cleaned, encoding='utf-8')
print('DONE')
