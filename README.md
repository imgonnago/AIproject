**AIproject**🚀
---
비판적 재평가 텍스트를 통한 planner Model 성능 개선 듀얼 시스템 VLA 모델 개발
---
**Planner** : OpenVLA 7B fine tuning + 4bit, Frozen, CPU inference 

**Actor** : Qwen2.5VL 3B + 4bit + LoRA + GRPO

**Simulator** : LIBERO-long
---
**Back Bone Model URL**
**- OpenVLA** : https://github.com/openvla/openvla.git
**- Qwen2.5VL** : https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl
**- LIBERO** : https://github.com/Lifelong-Robot-Learning/LIBERO.git
---
#github file structure
**<openvla_planner>**
- openvla inference code
  -> openvla inference만 하기 위한 코드. 전체 코드는 필요없음
  **cpu**를 사용하여 inference 할 것 이므로 cuda 사용하지 않음.
  transformer로 로드하면 됨.
  모델은 **openvla-7b-finetuned-libero-10** 로 로드
  
- **zeroMQ**
  -> actor로 출력물인 action token을 반환 하기위해 전송하는 코드.
  직접 구현해야함.
  
- action tokenizer
  -> openvla 의 action tokeinzer원본.
  확인하면서 코딩하기 위해 편의성으로 유지.

**<qwen_actor>**
- Qwen2.5VL
  -> qwen2.5vl 전체코드를 유지.
  토크나이저와 projection layer등을 구현하여 적용시켜야 하므로 전체 코드를 보면서 코딩하기 위함.
- **qwen_llm_tokenizer**
  -> LLM tokenizer에 openvla의 256개 action token을 추가.
- **projection_layer**
  -> openvla와 qwen2.5vl의 토크나이저 임베딩 공간이 달라서 이를 맞춰주기위한 layer.
  
  openvla의 4096차원 action 임베딩을 그냥 넣으면 차원이 안 맞음.
- **actoin tokenizer**
  -> qwen에 붙여줄 action tokenizer 구현.
  input(actoin token) -> projection layer
  
                 concat
  
  image, text -> qwen processor(LLM vocabulary에 openvla의 action token 256개 추가)

**<train>**
- GRPO로 학습할 수 있도록 train 코드를 구현해야 함.
