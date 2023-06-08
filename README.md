# opensw23-team_KSY

## Team Introduction
- 김소연(201911244)

## Topic Introduction
In-Domain GAN Inversion을 통해 Semantic Diffusion, Interpolation, Manipulation, Style Mixing을 수행할 수 있습니다. 
GAN Inversion은 입력 이미지와 유사한 결과 이미지를 얻을 수 있도록 하는 잠재벡터(latent vector)를 찾는 과정을 말합니다. 기존 inversion methods는 타겟 이미지를 픽셀 값에 의해 재구성하는데에 집중하여 기존 latent space의 semantic domain에서 inverted code를 배치시키는데 실패했습니다. 입력 이미지의 latent code를 추론하는 기능이 없기 때문에 이미지에 대한 조작을 적용하기 어려웠습니다. 반면 In-Domain GAN inversion은 입력 이미지를 재구성할 뿐만 아니라 semantically meaningful한 latent code로 invert가 가능하도록 하였습니다.

original github repository: https://github.com/genforce/idinvert_pytorch

## Results

**Semantic Diffusion**

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/172a9b72-fb56-4fc2-86df-2fc283f22268)
![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/b0d1be3e-26c2-48a3-b728-9e27cc3ad32b)
![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/a5a6cada-c3c4-4c90-b35d-de92778c48c1)

**Interpolation** 

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/8cf49173-ddda-41fb-b323-2154dc152314)
![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/0e6d84df-b8d4-46f8-a8c0-0a5ded4563ac)

**Manipulation**

expression

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/bc4b845f-1269-40ba-95e0-9085804a0439)

age

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/664c401b-56ca-4b0c-924d-25e216dddc8f)


**Style Mixing**

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/fc0758c2-ab6c-4d30-9a1e-3705f82d060b)


## Analysis

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/3a58b1d5-974a-44d3-a4e9-e106866e8262)
![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/ee668edb-f6f2-4a44-8a5e-1de968d10319)
Semantic Diffusion의 경우, 두 이미지의 얼굴의 방향이 서로 다르거나 앞머리가 있으 나쁜 결과가 발생합니다.

## Installation
1. 원하는 경로로 이동 후, git clone https://github.com/soxoy0181/opensw23-team_KSY.git
2. 다음 세가지를 다운로드 받은 후 models/pretrain 내에 위치한다.
- face_256x256_generator(https://drive.google.com/file/d/1SjWD4slw612z2cXa3-n38JwKZXqDUerG/view?usp=sharing)
- face_256x256_encoder(https://drive.google.com/file/d/1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO/view?usp=sharing)
- Perceptual Model(https://drive.google.com/file/d/1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y/view?usp=sharing)
3. cd opensw23-team_KSY
4. 다음을 입력(수행하기 전에는 examples 폴더는 비워진 상태여야 함)

**Inversion**
1) examples 폴더 안에 원하는 이미지를 넣는다.
2) test.list에는 원하는 이미지의 경로들을 한 줄에 하나씩 써준다.
3) python invert.py styleganinv_ffhq256 examples/test.list

**Semantic Diffusion**
1) examples 폴더 안에 원하는 이미지를 넣는다.
2) target.list에는 타겟 이미지의 경로들을 써주고, context.list에는 컨텍스트 이미지의 경로들을 써준다.
3) python diffuse.py styleganinv_ffhq256 examples/target.list examples/context.list

**Interpolation** 
1) 원하는 이미지들에 대해 Inversion을 수행한다.
2) python interpolate.py styleganinv_ffhq256 results/inversion/test results/inversion/test

**Manipulation**
1) 원하는 이미지들에 대해 Inversion을 수행한다.
2) python manipulate.py styleganinv_ffhq256 results/inversion/test boundaries/stylegan_ffhq256/expression.npy (expression 외에도 age, eyeglasses, gender, pose로 바꾸어 수행 가능)

**Style Mixing**
1) 원하는 이미지들에 대해 Inversion을 수행한다.
2) python mix_style.py styleganinv_ffhq256 results/inversion/test results/inversion/test

## Presentation
