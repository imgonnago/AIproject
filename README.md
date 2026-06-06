<h1>AIproject : 비판적 재평가 텍스트를 통한 planner Model 성능 개선 듀얼 시스템 VLA 모델 개발</h1> 

> OpenVLA(Planner)와 SmolVLM 500B(Actor)을 결합한 듀얼 시스템 VLA(Vision-Language-Action) 모델.
> Planner가 생성한 action을 Actor가 **비판적으로 재평가**하여 조작 성능을 개선하는 구조를 제안하고 구현함.

**Critical text And Dual System Vision Language Action Model(CADS-VLA)**

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

## 🔎소개 및 문제 정의

기존 단일 VLA 모델은 시각·언어 입력으로부터 곧장 action을 생성하지만,
생성한 action이 적절한지 **스스로 검토하는 단계**가 없음. 그 결과
장기 과제(long-horizon task)에서 한 번의 잘못된 판단이 전체 실패로 이어지기 쉬움.

이는 추측이 아니라 벤치마크에서 정량적으로 드러남. CoT-VLA 논문(Zhao et al., 2025)의
LIBERO 평가표를 보면, **동일한 조건(3 seeds × 500 episodes)**에서 평가한 모델들 중
재평가·검토 단계가 없는 단일 VLA들은 단기 과제에서는 높지만 Long suite에서만 50%대로 급락함.

**LIBERO 벤치마크 — Task 별 Success Rate (동일 평가 조건)**

| 모델 | 추론·검토 단계 | Spatial | Object | Goal | **Long** |
|------|:------------:|:-------:|:------:|:----:|:--------:|
| Diffusion Policy | 없음 | 78.3% | 92.5% | 68.3% | **50.5%** |
| Octo | 없음 | 78.9% | 85.7% | 84.6% | **51.1%** |
| OpenVLA-7B | 없음 | 84.7% | 88.4% | 79.2% | **53.7%** |
| **CoT-VLA-7B** | **있음** (visual CoT) | 87.5% | 91.6% | 87.6% | **69.0%** |

> 위 4개는 **CoT-VLA 논문 Table 1의 동일 표에서** 같은 조건으로 평가된 수치.
> 아키텍처가 달라도(autoregressive, diffusion) 검토 단계가 없으면 Long에서 50%대로 무너지는 반면,
> **중간 추론 단계(visual chain-of-thought)를 추가한 CoT-VLA는 같은 Long task에서 69.0%로,
> 다른 task와의 격차가 크게 줄어듦.** 변수는 "행동 전 검토 단계의 유무".

---

**→ 즉, "행동 전에 검토·교정하는 단계"의 유무가 long-horizon 성공률에 영향을 줌.**
본 프로젝트의 비판적 재평가 구조는 바로 이 검토 단계를, **별도 모델(Actor)을 통한
역할 분리** 방식으로 구현하려는 시도임.

> *(출처: Zhao et al., "CoT-VLA: Visual Chain-of-Thought Reasoning for VLA Models", arXiv:2503.22020, Table 1.
> Diffusion Policy·Octo·OpenVLA·CoT-VLA 모두 동일 논문에서 3 seeds × 500 episodes 조건으로 평가됨.)*

원인은 명확함 — 지도학습 기반 모방학습은 긴 시퀀스에서 오류 누적에 취약해,
초기의 사소한 실수가 과제 전체 실패로 이어지기 때문. 중간에 자신의 판단을
되짚는 단계가 이 누적을 끊어 매 순간 바로잡을 수 있음.

본 프로젝트는 이를 **역할 분리**로 해결.

- **Planner (OpenVLA)** — 1차 action을 빠르게 제안
- **Actor (SmolVLM 500B)** — Planner의 제안을 비판적 재평가 텍스트와 함께 검토하고 교정

이는 사람의 "생각 → 점검 → 실행" 흐름을 모방한 듀얼 시스템 구조.

---

이 프로젝트는 **모델 아키텍처와 파이프라인 구현을 완료**했으며, 학습은 자원(컴퓨팅·시간) 제약으로 다음 단계까지 진행함.

| 단계 | 상태 |
|------|------|
| 듀얼 시스템 아키텍처 설계 | ✅ 완료 |
| Planner(OpenVLA) 추론 파이프라인 | ✅ 완료 |
| Actor(SmolVLM 500B) 구성 (토크나이저 확장 · projection layer) | ✅ 완료 |
| ZeroMQ 기반 프로세스 간 통신 | ✅ 완료 |
| **SFT — Actor의 action token 출력 + 텍스트 생성 능력** | 🟡 일부 학습, **정성적 동작 확인 완료** |
| GRPO 강화학습 | ⬜ 미진행 (코드는 구현, 대규모 학습 미실행) |
| LIBERO-long 정량 평가 | ⬜ 미진행 (컴퓨팅 자원 제약) |

> 즉, **"끝까지 동작하는 시스템을 만들고, 핵심 학습 단계가 의도대로 작동함을 검증한"** 단계.
> 정량적 벤치마크 결과는 향후 과제임.

---
**Planner** : OpenVLA 7B fine tuning + 4bit, Frozen, CPU inference 

**Actor 1** : Qwen2.5VL 3B + 4bit + LoRA + GRPO

**Actor 2** : SmolVLM 500M + 4bit + LoRA + GRPO <- **(used)**

**Simulator** : LIBERO (libero_spatial)


## 🦴Back Bone Model URL

**- OpenVLA** : [openvla](https://github.com/openvla/openvla.git)

**- Qwen2.5VL** : [Qwen2.5VL](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_5_vl)

**- SmolVLM** : [SmolVLM](https://github.com/huggingface/smollm/tree/main)

**- LIBERO** : [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO.git)

## 🖼️아키텍쳐

![Figure1](https://github.com/user-attachments/assets/edbb073d-d40e-452a-873f-a4de54bc41b6)
- **1. Model Structure**

 모델의 전체적 구조를 간단하게 도식화 함. planner의 openvla 모델은 forzen으로 사용. actor의 backbone 모델은 LVM 모델 중 매우 가벼운 SmolVLM 500M을 사용. Planner는 action token id를 actor로 전달. actor는 이를 openvla embedding table로 받아 projection layer를 거쳐 차원을 맞춘 뒤 image와 text를 smolvlm의 processor를 거친 임베딩과 concat함. 이후 llm을 거쳐 나온 임베딩을 detoken 과정을 거쳐 텍스트로 표현되고 로봇이 액션. 로봇의 액션 성공 실패 여부로 reward를 받아 GRPO 학습 커리큘럼을 통해 학습됨. 
 
---
![Figure2](https://github.com/user-attachments/assets/15b2c0a8-c4bb-4968-bf58-09bde7475a99)
- **2. Model Process**

 모델의 전체적 프로세스를 그림으로 표현. planner는 이미지와 텍스트를 openvla 자체 llm과 vit로 처리한 뒤 action token 추론을 통해 action token id를 내보냄. actor는 이미지와 텍스트를 smolvlm 자체 llm과 vit로 처리. planner에서 받아온 action token은 openvla에서 가져온 action embedding table(frozen)을 통해 임베딩으로 변환. 이후 projection layer를 통해 smolvlm의 임베딩 차원으로 변환하여 processor와 concat함. 이때 smolvlm의 tokenizer에는 openvla의 256개 action token이 add 되어있음. 이를 과정을 거쳐 transformer에서 attention 연산을 한 뒤 디토크나이저에서 텍스트를, action token은 action vector의 형태로 로봇으로 들어가고 행동을 함. 
 
---
![Figure3](https://github.com/user-attachments/assets/cf47ba2e-f4df-4c42-9f32-b77c0ebf5a60)
- **3. Projection Layer**

  Openvla Embedding(4096) 차원을 Smolvlm embedding(960) 차원으로 맞춰주는 역할. projection의 구조는 LLaVA의 projection Layer를 기반으로 가져옴.

---
![Figure4](https://github.com/user-attachments/assets/e5adbf70-5ef6-4e22-94d6-8828ed1dc6be)
- **4. FLow Chart**

 모델의 플로우 차트. 모델에서의 전체적인 데이터의 흐름을 표현. planner와 actor의 구조는 앞서 설명한 과정과 같아서 생략. actor와 planner에서 사용하는 이미지, 텍스트 데이터들은 모두 LIBERO 환경에서 수집됨. zeroMQ를 통해 actor가 planner 서버로 넘기는 구조. 위에 actor action tokenizer init 과정에서 256개 액션 토큰을 openvla에서 가져와 add -> action embedding table -> resieze -> projection layer 과정으로 tokenizer를 초기화 시킴. 아래 GRPO trian 과정은 RLinf의 프레임워크를 반영. collect rollout 에서 libero 환경에서의 데이터들을 그룹 사이즈만큼 수집. 이 과정에서 비판적 텍스트와 action token을 추론하여 액션을 수행하고 그에 따른 reward와 loss등을 계산하고 가중치 업데이트를 compute 과정에서 수행. 이후 이를 통해 LoRA를 학습시킴. 
 
---
![Figure5](https://github.com/user-attachments/assets/1c0fbf23-fe96-475a-8827-3fd545224f3e)
- **5. Actor Tokenizer Process**

 구현한 actino tokenizer의 구조를 구체화. 각 데이터들은 각각 vision encoder, llm, projection layer를 통과하여 임베딩 형태로 변환되고, input merge와 action injection hook을 통해 같은 공간으로 투영되고 concat을 진행함. 이때 이미지와 텍스트는 smolvlm의 프로세서를 거침. 이후 트랜스포머를 통과하고 출력을 냄.
 
---
![Figure6](https://github.com/user-attachments/assets/f106f294-3e26-4129-ab98-1bde2140310e)
- **6. Suprevised Fine Tuning**
  
 바로 모델 학습으로 들어가면 모델이 텍스트 포멧과 액션토큰을 어떻게 내보내는지 알지 못함. GRPO 학습에서 계속 패널티를 받고, 수렴하지 못하는 문제 발생 가능. 그래서 SFT를 통해 기본적인 베이스 능력을 학습시킨 뒤 비판적 텍스트와 그에 따른 액션 토큰 수정을 하도록 하기 위함. 총 4개 스테이지로 구성되었고, 스테이지마다 순차적으로 학습.

---
**핵심 기술 과제와 해결**

| 문제 | 해결 | 상태 |
|------|------|------|
| OpenVLA(4096차원 action 임베딩)와 Smol 토크나이저의 임베딩 공간 불일치 | LLaVA projection layer를 참고한 **projection layer** 구현으로 차원 정합 | ✅ 완료 |
| Smol LLM이 action을 토큰으로 다루지 못함 | LLM vocabulary에 OpenVLA의 **action token 256개 추가** | ✅ 완료 |
| 두 모델이 독립 프로세스/환경에서 동작 | **ZeroMQ**로 프로세스 간 action token 중계 | ✅ 구현 |
| 제한된 자원에서 7B+500B 모델 동시 운용 | 4bit 양자화 · Planner Frozen · Actor LoRA로 메모리 효율화 | ✅ 완료 |
| Actor가 action token과 자연어를 함께 출력해야 함 | SFT로 두 출력 능력 학습 | 🟡 정성 확인 |

## 검증 결과 (Validation)

정량 벤치마크(success rate)는 미진행이나, 구현이 의도대로 동작함을 다음과 같이 정성적으로 확인함.

- **Actor의 action token 출력** — SFT 이후 Actor가 추가된 256개 action token을 정상적으로 생성함을 확인
- **텍스트 생성 능력** — action token 학습 후에도 자연어 출력 능력이 일부 유지됨 확인
  > action token 생성 능력 학습으로 인해 텍스트 생성 능력 일부 약화 
- **End-to-end 파이프라인** — Planner → ZeroMQ → Actor로 이어지는 데이터 흐름이 동작함을 확인

![work](https://github.com/user-attachments/assets/52948375-398b-4432-8eab-1c4049e69324)

> planner의 추론 action token을 학습 데이터로 사용하여 형식 학습. 성공적으로 7개 토큰 출력.
> 
> 정량 평가(LIBERO-long success rate, baseline 대비 비교)와 GRPO 학습은 충분한 컴퓨팅 자원 확보 후 진행할 향후 과제.

## 👷코드 구조 

**📁openvla_planner**

- **openvla inference code**
  
  > openvla inference만 하기 위한 코드.
  >zeroMQ와 통합하여 서버를 열어줌.
  >**cpu**를 사용하여 inference 할 것 이므로 cuda 사용하지 않음.
  >transformer로 로드하면 됨.
  >모델은 **openvla-7b-finetuned-libero-spatial** 로 로드

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
  > 
  > LLaVA의 projection layer를 참고하여 구현.
  >
  >  **LLaVA** : [LLaVA](https://github.com/haotian-liu/LLaVA/blob/main/llava/model/multimodal_projector/builder.py)

- **actor_model**

  > qwen에 4bit quantization + LoRA를 적용시키고 zeroMQ와 통합한 actor의 실행파일.

---

**📁SmolVLM_actor**

- **smol_action_tokenizer**

  > smolvlm LLM tokenizer에 openvla의 액션 토큰을 이식.
  > 모델 실행시 토크나이저를 초기화시킴.

- **smol_actor_model**

  > smol에 4bit quatization + LoRA + zeroMQ 적용.

- **smol_projection_layer**

  > openvla와 smolvlm의 토크나이저 임베딩 공간을 맞춰줌. 
  > LLaVA의 prijection layer를 참고함.

---

**📁train_file**

- **train**

  > main역할을 하는 파일임. GRPO와 LIBERO를 실행.

- **smol_train**

  > smolvlm train실행 파일.
  > GRPO를 RLinf를 통해 구현함.
  >
  > **RLinf** : [RLinf](https://github.com/RLinf/RLinf)
  >
  > colliect_rollout과 compute_grpo_loss 함수로 학습 진행.
  > 모델 실행시 해당 파일을 실행해야함.

- **smol_sft**

  > GRPO강화학습을 진행하기 전 SFT를 통해 모델이 사전 학습을 하도록 유도.
  > DeepSeek에서 사용한 방식.
  > 강화학습 모델의 성능이 더 좋아지고, 성능에도 긍정적 효과를 준다고 증명됨.
  > 여기선 모델의 비판적 텍스트 생성과 7개 액션 토큰 생성 능력을 sft를 통해 학습 시킴.

---

**📁SFT**

- **SFT**
  
  > GRPO 학습을 들어가기 전 사전 학습을 통해 actor에게 텍스트 포멧 형식을 학습시키기 위함.

---

**📁assets**

- **make_embeddings.py**

  > openvla action embedding 파라미터를 다운받는 파일.
  > 모델 실행시키기 전에 무조건 한 번 실행시켜야함.

---

**📁checkpoints**

- **sft**

 > SFT를 수행한 모델 체크포인트 파일.
 > train이 마음에 안 들때 해당 체크포인트로 다시 학습.
 > 해당 학습을 시킬 때 비전 인코더를 사용하지 않은 오류가 있음. 사용하지 않음

- **sft2**

  > SFT 수행한 체크포인트.
  > 비판적 텍스트와 액션 토큰을 추론하는 능력을 기본적으로 학습 시킴.
  > 해당 체크파일은 비전 인코더 사용 버전으로, 해당 파일 사용.

---

**📁logs**

- 학습시 나왔던 로그들을 저장해둠. 어떻게 학습이 되었는지 기록용


## ⏬환경 구성

```
#torch version (qwen)

pip install torch torchvision torchaudio \ --index-url https://download.pytorch.org/whl/cu124

pip install requirements_qwen.txt --no-deps

git clone https://github.com/Lifelong-Robot-Learning/LIBERO

#torch version (openvla)

pip install torch==2.12.0+cpu torchvision==0.27.0+cpu torchaudio==2.11.0+cpu --index-url https://download.pytorch.org/whl/cpu

pip install requirements_openvla.txt
```

SmolVLM 모델 실행시 미리 구성한 qwen 환경에서 실행해도 무방함. 

## 🏁실행 방법 (Getting Started)
---

- 모델을 실행하기 전 openvla_embeddings 파일이 필요함.openvla 환경에 진입해서 

```
git clone https://github.com/imgonnago/AIproject

conda activate openvla

python make_embeddings.py #embedding파일 생성.

python openvla_planner/openvla_inference_code.py

#actor 환경 진입 및 실행(SmolVLM도 qwen 환경 사용)

conda activate qwen

#파일 실행시 openvla를 먼저 실행한 뒤 zeroMQ 서버가 열리고 qwen을 실행해야함.

python train/train.py
```

## ⚙️기타 설정
---

모델 실행시 vram 사용량과 train log를 기록할 수 있는 코드.

```
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
```

vscode에서 계속 SSH서버가 끊어질 때, 우분투에서

`tmux new -s planner`

`conda activate openvla`

실행하기 전 프로젝트 폴더 안으로 이동해서 실행.

`python openvla_planner/openvla_inference_code.py`

`tmux new -s train`

`conda activate qwen`

마찬가지로 프로젝트 폴더 안으로 이동해서 실행.
```
#모델 실행 전 이 환경변수를 설정하면 vram에 도움됨.
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

`nvidia-smi --query-gpu=timestamp,memory.used --format=csv -l 1 >> vram_log.csv &`

`python train/smol_train.py 2>&1 | tee train_log.txt`s
