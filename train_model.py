"""
모델 학습 스크립트
Streamlit Cloud에서 자동으로 실행되어 모델을 생성합니다.
"""

import os
import joblib
from model import BidRatePredictor
from data_processor import DataProcessor

def train_and_save_model():
    """모델 학습 및 저장"""
    print("Model training started...")

    # 데이터 처리
    processor = DataProcessor()
    data = processor.process_pipeline('강원도및경기일부.xlsx')

    if data is None:
        print("Data processing failed")
        return False

    # 모델 학습
    predictor = BidRatePredictor(model_type='linear')
    predictor.train(data['X_train'], data['y_train'])

    # 모델 평가
    metrics = predictor.evaluate(data['X_test'], data['y_test'])

    # 모델 저장
    if not os.path.exists('models'):
        os.makedirs('models')

    model_data = {
        'model': predictor.model,
        'processor': processor,
        'metrics': metrics,
        'feature_columns': data['feature_columns']
    }

    joblib.dump(model_data, 'models/trained_model.pkl')
    print("Model saved: models/trained_model.pkl")

    return True

if __name__ == "__main__":
    # 모델이 없을 때만 학습
    if not os.path.exists('models/trained_model.pkl'):
        train_and_save_model()
    else:
        print("Existing model found.")