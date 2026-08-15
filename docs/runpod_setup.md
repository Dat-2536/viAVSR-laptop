## 1. Clone `main` mới nhất

```bash
git clone \
  --branch main \
  --single-branch \
  https://github.com/longdibanbao/viASVR-laptop.git \
  viAVSR-laptop

cd viAVSR-laptop
```

Xác nhận VIAVSR-4 đã nằm trong `main`:

```bash
git status -sb

git merge-base --is-ancestor 15fcee2 HEAD \
  && echo "VIAVSR-4 is included"
```

Sau đó tạo branch riêng:

```bash
git switch -c VIAVSR-5-official-sample-inference
```

## 2. Tạo Conda environment một lần

Từ repository root:

```bash
conda env create -f environment/environment.yml
conda activate viavsr
```

`environment.yml` đã được sửa thành `-e ..[dev]`, nên không cần chạy thêm `pip install -e ".[dev]"`.

Nếu máy đã có environment `viavsr`:

```bash
conda env update \
  -f environment/environment.yml \
  --prune

conda activate viavsr
```

Không chạy cả `env create` và `env update`; chọn theo tình trạng của máy.

## 3. Kiểm tra Python và CUDA

```bash
python -c \
'import torch, viavsr; print("viavsr:", viavsr.__file__); print("torch:", torch.__version__); print("CUDA:", torch.cuda.is_available()); print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")'
```

Đối với full VIAVSR-4/5 run cần:

```text
torch: 2.7.1
CUDA: True
```

Nếu laptop không có NVIDIA GPU, thành viên vẫn có thể chạy unit tests và phát triển logic như bình thường, còn full model/sample inference có thể chạy trên Runpod A5000.

## 4. Tải tokenizer Vietnamese

```bash
python scripts/fetch_tokenizer_assets.py \
  --config configs/vietnamese_avsr.yaml
```

Checksum cần phải xuất ra đúng như sau:

```text
model:
21ca39e799b64044d75edccd9016fac0315e64f89bdd43fbd3089607dceb9d64

units:
ea7b25e67a302305ffdb59909419c08822b3607a6b03871adef2bcb9f6ebec25
```

## 5. Chạy test

```bash
python -m pytest -q
```

Baseline hiện tại đã đúng hết như sau:

```text
35 passed, 5 skipped
```

nên không được có `failed`, nếu có phải hỏi lại lead.

## 6. Chuẩn bị Hugging Face token

Mỗi thành viên phải tạo token Read-Only riêng trên Hugging Face, và không chia sẻ token của nhau.

Nhập token mà không ghi vào shell history:

```bash
read -rsp "Hugging Face token: " HF_TOKEN
export HF_TOKEN
echo
```

Nếu dùng Runpod PyTorch template như sau:

```bash
unset HF_HUB_ENABLE_HF_TRANSFER
```

Xác nhận:

```bash
python -c \
'import os; print("HF_TOKEN configured:", bool(os.environ.get("HF_TOKEN")))'
```

## 7. Chạy VIAVSR-4 preflight trước khi code VIAVSR-5

```bash
python scripts/check_model_assets.py \
  --config configs/vietnamese_avsr.yaml
```

Phải xuất ra như sau:

```text
Model loaded: PASSED
Tokenizer loaded: PASSED
SentencePiece vocabulary: 2048
ASR tokenizer vocabulary: 2057
Model output vocabulary: 2057
Vocabulary compatibility: PASSED
Vietnamese round-trip: PASSED
Device: cuda
```

Nếu preflight không pass thì chưa nên bắt đầu inference media.

## 8. Quy tắc triển khai VIAVSR-5

VIAVSR-5 nên tái sử dụng loader hiện tại:

```python
from viavsr.inference import (
    load_model_assets_config,
    load_vietnamese_avsr_assets,
)
```

Không được:

- viết loader checkpoint mới trong script;
- hard-code `.cuda()`;
- thay tokenizer bằng `unigram5000`;
- đưa `HF_TOKEN` vào YAML;
- triển khai preprocessing laptop/Sprint 2;
- chỉnh model architecture để ép sample chạy.

Reusable inference logic đặt tại:

```text
src/viavsr/inference/
```

Script CLI chỉ orchestrate tại:

```text
scripts/
```

Hiện `main` chưa có official-sample inference entrypoint; đó chính là phần thành viên VIAVSR-5 cần triển khai.

## 9. Mỗi phiên làm việc sau này

Local:

```bash
conda activate viavsr
cd /path/to/viAVSR-laptop
```

Runpod:

```bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate viavsr
cd /workspace/viAVSR-laptop
unset HF_HUB_ENABLE_HF_TRANSFER
```

Không commit:

```text
HF token
model cache
tokenizer binaries
samples lớn
outputs/
logs
```