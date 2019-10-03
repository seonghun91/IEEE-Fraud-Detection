## IEEE-Fraud-Detection  

대회 목표 : detect fraud from customer transactions  
대회 기간 : '19/7/16 ~ '19/10/1    
대회 결과 : 11th (4,161 팀)  
멤버 : 권순환, 김경환, 김연민, 김현우, 정성훈  

**저는 EDA와 Feature Engineering을 주로 진행했습니다.**    

### 1. EDA (Exploratory Data Analysis)  

- 100_Base EDA : 데이터 변수별 의미를 파악하기 위한 대략적인 분석  
- 200_Find_Same_Users : 대회 초기 같은 유저를 추출할 수 있다면 많은 피쳐를 뽑아낼 수 있을 것이라 판단.  
 이에 따른 Same User 뽑아내기 위한 시도(train/test 분포 차이로 인해 실제 적용은 x)
- 300_multivarate_analysis : 변수간의 분포 파악
- 400_time_series_anaylsis : card/addr 등의 시간 별 분포를 파악
- 500_ideas : 이외 feature에 적용해볼만한 다양한 가설들을 적용

### 2. FE (Feature Engineering) - 정리 중  

- (100~500) : EDA 기반 feature들
- (600~900) : Feature Engineering 새로운 변수들 시도(clustering 등)

### 3. Modeling

- 100_Single_lightgbm_model : 현우님 baseline 바탕으로 feature들 추가하여 실험한 Single model  
- 800_adversarial_validation