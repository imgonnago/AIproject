<h1>AIproject : 비판적 재평가 텍스트를 통한 planner Model 성능 개선 듀얼 시스템 VLA 모델 개발</h1> 


<a href="https://pytorch.org/get-started/locally/">
  <img src="https://img.shields.io/badge/PYTORCH-Qwen%202.6.0%20cu124-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://pytorch.org/get-started/locally/">
  <img src="https://img.shields.io/badge/PYTORCH-openvla%202.12.0%20%2Bcpu%20-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://www.python.org/">
  <img src="https://img.shields.io/badge/Python-3.10-brightgreen?style=flat-square&label=Python&labelColor=%23eeeeee&color=%2355adf4"height="40"/>
</a>

---
**Planner** : OpenVLA 7B fine tuning + 4bit, Frozen, CPU inference 

**Actor 1** : Qwen2.5VL 3B + 4bit + LoRA + GRPO

**Actor 2** : SmolVLM 500B + 4bit + LoRA + GRPO <- (use)

**Simulator** : LIBREO-Long

---

**Back Bone Model URL**
---
**- OpenVLA** : [openvla](https://github.com/openvla/openvla.git)

**- Qwen2.5VL** : [Qwen2.5VL](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)

**- SmolVLM** : [SmolVLM](https://github.com/huggingface/smollm/tree/main)

**- LIBERO** : [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO.git)


**github file structure**
---

**📁openvla_planner**

- **openvla inference code**
  
  > openvla inference만 하기 위한 코드.
  >zeroMQ와 통합하여 서버를 열어줌.
  >**cpu**를 사용하여 inference 할 것 이므로 cuda 사용하지 않음.
  >transformer로 로드하면 됨.
  >모델은 **openvla-7b-finetuned-libero-10** 로 로드

- **action tokenizer**
  
  > openvla 의 action tokeinzer원본.
  >확인하면서 코딩하기 위해 편의성으로 유지.

---

**📁qwen_actor**

- **actor_action_tokenizer**
  
  > LLM tokenizer에 openvla의 256개 action token을 추가.
  > planner의 임베딩 테이블을 가져오고 projection layer를 사용하여 qwen과 차원을 맞춰줌.
  > Qwen processor와 concat까지 진행.
  >**setpu**과 **forward** 함수를 보면 됨.
  
- **projection_layer**
  
  > openvla와 qwen2.5vl의 토크나이저 임베딩 공간이 달라서 이를 맞춰주기위한 layer.
  > openvla의 4096차원 action 임베딩을 그냥 넣으면 차원이 안 맞음
  > LLaVA의 projection layer를 참고하여 구현.

  **LLaVA** : [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)

- **actor_model**

  > qwen에 4bit quantization + LoRA를 적용시키고 zeroMQ와 통합한 actor의 실행파일.

---

**📁SmolVLM_actor**

- **smol_action_tokenizer**

  > smolvlm LLM tokenizer에 openvla 256개 action token 추가.
  > qwen_actor의 기능들과 같음.

- **smol_actor_model**

  > smol에 4bit quatization + LoRA + zeroMQ 적용.

- **smol_projection_layer**

  > openvla와 smolvlm의 토크나이저 임베딩 공간을 맞춰줌. 
  > LLaVA의 prijection layer를 참고함.

---

**📁train_file**

- **train**

  > main역할을 하는 파일임. GRPO와 LIBERO를 실행.
  > 보상함수를 설계 해야함.

- **smol_train**

  > smolvlm train실행 파일.
  > 모델 실행시 해당 파일을 실행해야함.
  
---

**📁assets**

- **make_embeddings.py**

  > openvla action embedding 파라미터를 다운받는 파일.
  > 모델 실행시키기 전에 무조건 한 번 실행시켜야함.

PYTORCH VERSION (나머지는 requirements file 참고)
---
torch version (qwen)

`pip install torch torchvision torchaudio \ --index-url https://download.pytorch.org/whl/cu124`

torch version (openvla)

`pip install torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu`

SmolVLM 모델 실행시 미리 구성한 qwen 환경에서 실행해도 무방함. 

**모델 실행**
---
- 모델을 실행하기 전 openvla_embeddings 파일이 필요함.openvla 환경에 진입해서 

`python make_embeddings.py` embedding파일 생성.

- planner와 actor의 환경이 분리되어있어, 터미널 두 개를 사용함. 터미널1 에는 openvla, 터미널2 에는 qwen 환경을 진입.

```
conda actiavte openvla

conda activate qwen
```

각 터미널에서 파일 실행.

```
python openvla_planner/openvla_inference_code.py

#파일 실행시 openvla를 먼저 실행한 뒤 zeroMQ 서버가 열리고 qwen을 실행해야함.

python train/train.py

**기타 설정**
---
모델 실행시 vram 사용량과 train log를 기록할 수 있는 코드.

#vram_log

nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >> vram_log.csv &

#train_log

python train/train.py >> train_log.txt 2>&1  or  python train/smol_train.py >> train_log.txt 2>&1

"""
터미널에 입력하면 nvidia 프로세서 정보와 vram 사용량 gpu사용량을 볼 수 있는 코드.

숫자를 바꾸면 해당 초 마다 사용량을 볼 수 있음.
"""

watch -n 0.5 nvidia-smi

#GPU가 어떤 프로세스를 사용하는지 확인할 수 있는 코드.

nvidia-smi pmon -c 1
