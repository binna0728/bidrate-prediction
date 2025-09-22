# 사정율 예측 시스템 프로젝트

## 프로젝트 개요
- **목적**: 100번이동, 10번이동, 3번이동 데이터로 사정율 예측
- **데이터**: 강원도및경기일부.xlsx
- **목표 성능**: R² > 0.5

## 기술 스택

### 머신러닝
- Linear Regression (기본)
- XGBoost, LightGBM, CatBoost (부스팅)
- Random Forest (앙상블)

### 딥러닝
- DNN (Deep Neural Network)
- LSTM (시계열)
- Transformer (최신)

### 프레임워크
- PyTorch (딥러닝)
- Scikit-learn (머신러닝)
- Streamlit (웹앱)

## 프로젝트 구조

```
predict_model/
├── data/
│   └── 강원도및경기일부.xlsx
├── models/
│   └── (학습된 모델 저장)
├── notebooks/
│   ├── analysis.ipynb
│   ├── advanced_models.ipynb
│   └── transformer_models.ipynb
├── app.py (Streamlit 웹앱)
├── train_model.py (모델 학습)
├── requirements.txt
└── CLAUDE.md (이 문서)
```

## 실행 방법

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows

# 패키지 설치
pip install -r requirements.txt
```

### 2. 모델 학습
```bash
python train_model.py
```

### 3. 웹앱 실행
```bash
streamlit run app.py
```

### 4. Jupyter Notebook
```bash
jupyter notebook analysis.ipynb
```

## 모델 성능

| 모델 | R² Score | RMSE |
|-----|----------|------|
| Linear Regression | 0.34 | 0.60 |
| XGBoost | 0.40 | 0.57 |
| DNN | TBD | TBD |

## 배포

### Streamlit Cloud
1. GitHub 저장소 생성
2. https://share.streamlit.io 접속
3. GitHub 저장소 연결
4. Deploy 클릭

### Docker
```bash
docker build -t bidrate-predictor .
docker run -p 8501:8501 bidrate-predictor
```

## GPU 사용 (선택사항)

GPU가 있는 경우:
```bash
# PyTorch GPU 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 주요 파일 설명

- `app.py`: Streamlit 웹 애플리케이션
- `train_model.py`: 모델 학습 스크립트
- `data_processor.py`: 데이터 전처리
- `model.py`: 모델 정의
- `requirements.txt`: 필요 패키지 목록

## 문제 해결

### 한글 인코딩 문제
```python
# 파일 읽기 시
pd.read_excel('파일명.xlsx', encoding='utf-8')
```

### 메모리 부족
- 배치 크기 줄이기
- 데이터 샘플링 사용

## 개발 현황

- [x] 데이터 로드 및 전처리
- [x] Linear Regression 모델
- [x] XGBoost 모델
- [x] Streamlit 웹앱
- [x] GitHub 저장소 생성 및 push 완료
- [ ] 딥러닝 모델 최적화
- [ ] 배포

## WSL Ubuntu 작업 환경 설정

### 1. 저장소 클론
```bash
git clone https://github.com/binna0728/bidrate-prediction.git
cd bidrate-prediction
```

### 2. Python 환경 설정
```bash
# Python3 및 pip 설치 확인
python3 --version
sudo apt update
sudo apt install python3-pip python3-venv -y

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate
```

### 3. 패키지 설치
```bash
# pip 업그레이드
pip install --upgrade pip

# 필요 패키지 설치
pip install -r requirements.txt
```

### 4. 모델 학습 실행
```bash
# 모델 학습
python train_model.py

# Streamlit 앱 실행 (포트: 8501)
streamlit run app.py
```

### 5. Jupyter Notebook 실행
```bash
# Jupyter 실행
jupyter notebook --no-browser --port=8888

# 브라우저에서 접속: http://localhost:8888
```

## Windows와 WSL 간 작업

### Windows에서 작업한 내용을 WSL로 가져오기
```bash
# WSL에서
cd ~/bidrate-prediction
git pull origin main
```

### WSL에서 작업한 내용을 Windows로 보내기
```bash
# WSL에서
git add .
git commit -m "Update from WSL"
git push origin main

# Windows에서
git pull origin main
```

## 주의사항
- WSL과 Windows 간 줄바꿈 문자 차이 (LF vs CRLF)
- 경로 구분자 차이 (/ vs \)
- 파일 권한 차이

## 라이선스
Private Project