#!/usr/bin/env python
# coding: utf-8

from typing import Iterable, Tuple, Optional, Pattern
from tqdm import tqdm
import re
import fasttext

NICE_CHARS = re.compile("""[^\u0000-\u007F
\u0080-\u00FF
\u0100-\u017F
\u0180-\u024F
\u02B0-\u02FF
\u1E00-\u1EFF
\u2000-\u206F
\u2070-\u209F
\u20A0-\u20CF
\u20D0-\u20FF
\u2100-\u214F
\u2150-\u218F
\u2190-\u21FF
\u2200-\u22FF
\u2300-\u23FF
\u2400-\u243F
\u2440-\u245F
\u2460-\u24FF
\u2500-\u257F
\u2580-\u259F
\u25A0-\u25FF
\u2600-\u26FF
\u2700-\u27BF
\u27C0-\u27EF
\u27F0-\u27FF
\u2900-\u297F
\u2980-\u29FF
\u2A00-\u2AFF
\u2B00-\u2BFF
\u2C60-\u2C7F
\u2E00-\u2E7F
\uA720-\uA7FF
\uAB30-\uAB6F
\U0001D100-\U0001D1FF
\U0001D400-\U0001D7FF
\U0001F000-\U0001F02F
\U0001F030-\U0001F09F
\U0001F0A0-\U0001F0FF
\U0001F100-\U0001F1FF
\U0001F300-\U0001F5FF
\U0001F600-\U0001F64F
\U0001F650-\U0001F67F
\U0001F680-\U0001F6FF
\U0001F700-\U0001F77F
\U0001F780-\U0001F7FF
\U0001F800-\U0001F8FF
\U0001F900-\U0001F9FF
\U0001FA00-\U0001FA6F
\U0001FA70-\U0001FAFF]""", re.X)
# \uFFF0-\uFFFF


def filter_lines(fn: str,
                 regex: Pattern[str],
                 fn_len: Optional[int] = None,
                 ) -> Iterable[Tuple[Tuple[str, ...], str]]:
    with open(fn) as fh:
        for line in tqdm(fh, total=fn_len):
            matches = tuple(re.finditer(regex, line))
            if matches:
                # yield (tuple(x.group(0) for x in matches), line)
                pass
            else:
                yield line.strip()


if __name__ == "__main__":
    import sys
    model = fasttext.load_model("fasttext_models/lid.176.bin")
    with open(sys.argv[2], "w") as fout:
        for line in filter_lines(sys.argv[1], NICE_CHARS, sum(1 for _ in open(sys.argv[1]))):
            prediction = model.predict(line, k=2)
            if not (prediction[0][0] != "__label__sv" or prediction[1][0] - prediction[1][1] < 0.95):
                print(line, file=fout)
