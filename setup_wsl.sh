#!/bin/bash

echo "======================================"
echo "사정율 예측 시스템 WSL 환경 설정"
echo "======================================"

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python 버전 확인
echo -e "${YELLOW}1. Python 버전 확인...${NC}"
if ! command -v python3 &> /dev/null; then
    echo "Python3이 설치되어 있지 않습니다. 설치를 시작합니다..."
    sudo apt update
    sudo apt install -y python3 python3-pip python3-venv
fi
python3 --version

# 가상환경 생성
echo -e "${YELLOW}2. Python 가상환경 생성...${NC}"
if [ -d "venv" ]; then
    echo "기존 가상환경이 있습니다. 활성화합니다..."
else
    python3 -m venv venv
    echo "새 가상환경을 생성했습니다."
fi

# 가상환경 활성화
source venv/bin/activate

# pip 업그레이드
echo -e "${YELLOW}3. pip 업그레이드...${NC}"
pip install --upgrade pip

# 패키지 설치
echo -e "${YELLOW}4. 필요 패키지 설치...${NC}"
pip install -r requirements.txt

# 데이터 디렉토리 생성
echo -e "${YELLOW}5. 디렉토리 구조 확인...${NC}"
mkdir -p data
mkdir -p models
mkdir -p notebooks

# 설치 확인
echo -e "${YELLOW}6. 설치된 주요 패키지 확인...${NC}"
python -c "import pandas; print(f'pandas: {pandas.__version__}')"
python -c "import numpy; print(f'numpy: {numpy.__version__}')"
python -c "import sklearn; print(f'scikit-learn: {sklearn.__version__}')"
python -c "import xgboost; print(f'xgboost: {xgboost.__version__}')"
python -c "import streamlit; print(f'streamlit: {streamlit.__version__}')"

echo -e "${GREEN}======================================"
echo -e "설치 완료!"
echo -e "======================================${NC}"
echo ""
echo "다음 명령어로 실행할 수 있습니다:"
echo "  1. 모델 학습: python train_model.py"
echo "  2. 웹앱 실행: streamlit run app.py"
echo "  3. Jupyter 실행: jupyter notebook"
echo ""
echo "가상환경 활성화: source venv/bin/activate"