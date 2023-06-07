# opensw23-team_KSY

## Team Introduction
- 김소연(201911244)

## Topic Introduction
In-Domain GAN Inversion을 통해 Inversion, Semantic Diffusion, Interpolation, Manipulation, Style Mixing을 수행할 수 있습니다. 
GAN Inversion은 입력 이미지와 유사한 결과 이미지를 얻을 수 있도록 하는 latent vector를 찾는 과정을 말합니다. 기존 inversion methods는 타겟 이미지를 픽셀 값에 의해 재구성하는데에 집중하여 기존 latent space의 semantic domain에서 inverted code를 배치시키는데 실패했습니다. 반면 In-Domain GAN inversion은 입력 이미지를 재구성할 뿐만 아니라 semantically meaningful한 latent code로 invert가 가능하도록 하였습니다.

original github repository: https://github.com/genforce/idinvert_pytorch

## Results
**Inversion**

![000006_ori](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/02b4201d-8576-4deb-a282-851aa5401540)
![000006_enc](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/5f6a49c0-79d9-468d-83ba-f94d9d083cbf)
![000006_inv](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/6b70623a-a59b-4daa-ac6d-0bdf5f06b98d)

**Semantic Diffusion**

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/c8414aab-128a-44a9-8210-43b79da8c0d6)

![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/f494022c-973f-4192-951d-86aac90a61bd)

**Interpolation** 
![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/f2a64094-cbb8-4d92-9bd0-26769cf7446c)

**Manipulation**(expression)
![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/00f55970-9b02-4314-8f68-96c3a50bbc4a)

**Style Mixing**
![image](https://github.com/soxoy0181/opensw23-team_KSY/assets/127181364/c0b152da-c484-4bc2-96ff-4216288b9a33)


## Analysis/Visualization

## Installation
1. face_256x256_generator(https://drive.google.com/file/d/1SjWD4slw612z2cXa3-n38JwKZXqDUerG/view?usp=sharing), face_256x256_encoder(https://drive.google.com/file/d/1gij7xy05crnyA-tUTQ2F3yYlAlu6p9bO/view?usp=sharing), Perceptual Model(https://drive.google.com/file/d/1qQ-r7MYZ8ZcjQQFe17eQfJbOAuE3eS0y/view?usp=sharing)을 다운로드 받은 후 models/pretrain 내에 위치한다.
2. 원하는 경로로 이동 후, git clone https://github.com/soxoy0181/opensw23-team_KSY.git
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
