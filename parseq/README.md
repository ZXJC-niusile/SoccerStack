# PARSeq

Scene Text Recognition model, vendored from [baudm/parseq](https://github.com/baudm/parseq).

Only the PARSeq model is included; other models (ABINet, CRNN, TRBA, ViTSTR) and training/testing scripts have been removed.

## Usage

Install as editable package:

```bash
pip install -e ./parseq
```

Load from checkpoint:

```python
from strhub.models.utils import load_from_checkpoint
model = load_from_checkpoint('parseq/models/soccernet_parseq.ckpt')
```
