<h1>AIproject : 비판적 재평가 텍스트를 통한 planner Model 성능 개선 듀얼 시스템 VLA 모델 개발</h1> 


<a href="https://pytorch.org/get-started/locally/"style="text-decoration: none;">
  <img src="https://img.shields.io/badge/PYTORCH-Qwen%202.6.0%20cu124-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://pytorch.org/get-started/locally/"style="text-decoration: none;">
  <img src="https://img.shields.io/badge/PYTORCH-openvla%202.12.0%20%2Bcpu%20-brightgreen?style=flat-square&label=PYTORCH&labelColor=%23eeeeee&color=%23d63f3a" height="40"/>
</a>
&nbsp;
<a href="https://www.python.org/"style="text-decoration: none;">
  <img src="https://img.shields.io/badge/Python-3.10-brightgreen?style=flat-square&label=Python&labelColor=%23eeeeee&color=%2355adf4"height="40"/>
</a>

---
**Planner** : OpenVLA 7B fine tuning + 4bit, Frozen, CPU inference 

**Actor** : Qwen2.5VL 3B + 4bit + LoRA + GRPO

**Simulator** : LIBREO-Long

---

**Back Bone Model URL**
---
**- OpenVLA** : [openvla](https://github.com/openvla/openvla.git)

**- Qwen2.5VL** : [Qwen2.5VL](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)

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
  >planner의 임베딩 테이블을 가져오고 projection layer를 사용하여 qwen과 차원을 맞춰줌.
  >Qwen processor와 concat까지 진행.
  >**setpu**과 **forward** 함수를 보면 됨.
  
- **projection_layer**
  
  > openvla와 qwen2.5vl의 토크나이저 임베딩 공간이 달라서 이를 맞춰주기위한 layer.
  > openvla의 4096차원 action 임베딩을 그냥 넣으면 차원이 안 맞음
  > LLaVA의 projection layer를 참고하여 구현.

  **LLaVA** : [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)

- **actor_model**

  > qwen에 4bit quantizaton + LoRA를 적용시키고 zeroMQ와 통합한 actor의 실행파일.

---

**📁train_file**

- **train**
  > main역할을 하는 파일임. GRPO와 LIBERO를 실행.
  > 보상함수를 설계 해야함.
  
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

`pip install torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu`
