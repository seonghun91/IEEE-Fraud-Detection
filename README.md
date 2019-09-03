## IEEE-Fraud-Detection  

대회 목표 : detect fraud from customer transactions  
대회 기간 : '19/7/16 ~ '19/10/1    
대회 결과 : 11th (4,161 팀)  
멤버 : 권순환, 김경환, 김연민, 김현우, 정성훈  

**저는 EDA와 Feature Engineering을 주로 진행했습니다.  
코드와 분석을 활용하시는 것은 좋으나 삽질이 대부분이고, 틀린 내용이 다분할수 있으니 참고만 부탁드립니다.**  

### EDA (Exploratory Data Analysis)  

- 100_Base EDA : 데이터 변수별 의미를 파악하기 위한 대략적인 분석  
- 200_Find_Same_Users : 대회 초기 같은 유저를 추출할 수 있다면 많은 피쳐를 뽑아낼 수 있을 것이라 판단. 이에 따른 Same User 뽑아내기 위한 시도(하지만 실제 적용은 못했습니다.)
- 300_multivarate_analysis : 변수간 관계를 통해 분포 파악
- 400_time_series_anaylsis : card/addr 등의 시간 별 분포를 파악
- 500_ideas : 이외 feature에 적용해볼만한 다양한 가설들을 적용

### FE (Feature Engineering)  

- (100~500) : EDA 넘버링과 매핑되게 진행
- (600~900) : Feature Engineering 새로운 변수들 시도(clustering, LDA 등등)

### Model  

- 100_Single_lightgbm_model : 현우님 baseline 잡아두신것을 바탕으로 feature들 추가하여 만든 Single model  
- 800_adversarial_validation
- 900_prevent_shakeup : 대회 마지막 shakeup 방지를 위한 아이디어
