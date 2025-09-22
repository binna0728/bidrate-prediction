import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from data_processor import DataProcessor

class BidRatePredictor:
    """사정율 예측 모델 클래스"""

    def __init__(self, model_type='linear'):
        self.model_type = model_type
        self.model = None
        self.best_params = None
        self.metrics = {}
        self.feature_importance = None
        self.processor = DataProcessor()

        # 모델 선택
        self.models = {
            'linear': LinearRegression(),
            'ridge': Ridge(),
            'lasso': Lasso(),
            'random_forest': RandomForestRegressor(random_state=42),
            'gradient_boost': GradientBoostingRegressor(random_state=42)
        }

    def train(self, X_train, y_train, optimize=False):
        """모델 학습"""
        print(f"\n[TARGET] {self.model_type.upper()} 모델 학습 중...")

        if optimize and self.model_type in ['ridge', 'lasso', 'random_forest']:
            # 하이퍼파라미터 최적화
            self.model = self._optimize_hyperparameters(X_train, y_train)
        else:
            # 기본 모델 학습
            self.model = self.models[self.model_type]
            self.model.fit(X_train, y_train)

        print("[OK] 모델 학습 완료!")
        return self.model

    def _optimize_hyperparameters(self, X_train, y_train):
        """하이퍼파라미터 최적화"""
        print("[CONFIG] 하이퍼파라미터 최적화 중...")

        param_grids = {
            'ridge': {
                'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
            },
            'lasso': {
                'alpha': [0.001, 0.01, 0.1, 1, 10]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        }

        if self.model_type in param_grids:
            grid_search = GridSearchCV(
                self.models[self.model_type],
                param_grids[self.model_type],
                cv=5,
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            self.best_params = grid_search.best_params_
            print(f"[OK] 최적 파라미터: {self.best_params}")
            return grid_search.best_estimator_

        return self.models[self.model_type].fit(X_train, y_train)

    def predict(self, X):
        """예측"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. 먼저 train()을 실행하세요.")

        predictions = self.model.predict(X)
        return predictions

    def predict_with_confidence(self, X, confidence_level=0.95):
        """신뢰구간과 함께 예측"""
        predictions = self.predict(X)

        # 간단한 신뢰구간 계산 (실제로는 더 정교한 방법 필요)
        std_error = np.std(predictions) * 0.1  # 임시 표준오차
        z_score = 1.96 if confidence_level == 0.95 else 2.58  # 95% or 99%

        lower_bound = predictions - z_score * std_error
        upper_bound = predictions + z_score * std_error

        return {
            'predictions': predictions,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'confidence_level': confidence_level
        }

    def evaluate(self, X_test, y_test):
        """모델 평가"""
        print("\n[DATA] 모델 평가 중...")

        y_pred = self.predict(X_test)

        # 평가 지표 계산
        self.metrics = {
            'r2_score': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred) * 100
        }

        # 교차 검증
        cv_scores = cross_val_score(self.model, X_test, y_test, cv=5, scoring='r2')
        self.metrics['cv_mean'] = cv_scores.mean()
        self.metrics['cv_std'] = cv_scores.std()

        # 결과 출력
        print("\n[CHART] 평가 결과:")
        print(f"  - R² Score: {self.metrics['r2_score']:.4f}")
        print(f"  - RMSE: {self.metrics['rmse']:.4f}")
        print(f"  - MAE: {self.metrics['mae']:.4f}")
        print(f"  - MAPE: {self.metrics['mape']:.2f}%")
        print(f"  - CV Score: {self.metrics['cv_mean']:.4f} (±{self.metrics['cv_std']:.4f})")

        return self.metrics

    def get_feature_importance(self, feature_names):
        """특성 중요도"""
        if hasattr(self.model, 'feature_importances_'):
            # Tree 기반 모델
            importances = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear 모델
            importances = np.abs(self.model.coef_)
        else:
            return None

        # 중요도 정렬
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)

        self.feature_importance = feature_importance
        return feature_importance

    def plot_predictions(self, y_test, y_pred, save_path=None):
        """예측 결과 시각화"""
        plt.figure(figsize=(12, 4))

        # 1. 실제 vs 예측
        plt.subplot(1, 3, 1)
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('실제값')
        plt.ylabel('예측값')
        plt.title('실제 vs 예측')

        # 2. 잔차 플롯
        plt.subplot(1, 3, 2)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('예측값')
        plt.ylabel('잔차')
        plt.title('잔차 플롯')

        # 3. 잔차 분포
        plt.subplot(1, 3, 3)
        plt.hist(residuals, bins=30, edgecolor='black')
        plt.xlabel('잔차')
        plt.ylabel('빈도')
        plt.title('잔차 분포')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"[OK] 그래프 저장: {save_path}")

        return plt.gcf()

    def save_model(self, path='models/'):
        """모델 저장"""
        if not os.path.exists(path):
            os.makedirs(path)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{path}{self.model_type}_model_{timestamp}.pkl"

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'metrics': self.metrics,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'processor': self.processor
        }

        joblib.dump(model_data, filename)
        print(f"[OK] 모델 저장 완료: {filename}")
        return filename

    def load_model(self, filename):
        """모델 로드"""
        model_data = joblib.load(filename)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.metrics = model_data.get('metrics', {})
        self.best_params = model_data.get('best_params')
        self.feature_importance = model_data.get('feature_importance')
        self.processor = model_data.get('processor', DataProcessor())
        print(f"[OK] 모델 로드 완료: {filename}")
        return self.model

    def compare_models(self, X_train, y_train, X_test, y_test):
        """여러 모델 비교"""
        print("\n[COMPARE] 모델 비교 중...")
        results = {}

        for model_name in self.models.keys():
            print(f"\n--- {model_name.upper()} ---")
            self.model_type = model_name
            self.train(X_train, y_train)
            metrics = self.evaluate(X_test, y_test)
            results[model_name] = metrics

        # 결과 정리
        comparison_df = pd.DataFrame(results).T
        comparison_df = comparison_df.sort_values('r2_score', ascending=False)

        print("\n[DATA] 모델 비교 결과:")
        print(comparison_df[['r2_score', 'rmse', 'mae']])

        # 최고 모델 선택
        best_model = comparison_df.index[0]
        print(f"\n[BEST] 최고 성능 모델: {best_model.upper()}")

        return comparison_df


def main():
    """메인 실행 함수"""
    print("="*60)
    print("[TARGET] 사정율 예측 모델 학습")
    print("="*60)

    # 1. 데이터 처리
    processor = DataProcessor()
    data = processor.process_pipeline()

    if data is None:
        print("[ERROR] 데이터 처리 실패")
        return

    # 2. 모델 학습 및 평가
    predictor = BidRatePredictor(model_type='linear')

    # 단일 모델 학습
    predictor.train(data['X_train'], data['y_train'], optimize=False)

    # 평가
    metrics = predictor.evaluate(data['X_test'], data['y_test'])

    # 특성 중요도
    if data['feature_columns']:
        importance = predictor.get_feature_importance(data['feature_columns'])
        if importance is not None:
            print("\n[DATA] 상위 5개 중요 특성:")
            print(importance.head())

    # 예측 시각화
    y_pred = predictor.predict(data['X_test'])
    predictor.plot_predictions(data['y_test'], y_pred, save_path='prediction_results.png')

    # 모델 저장
    model_path = predictor.save_model()

    # 3. 여러 모델 비교 (옵션)
    print("\n" + "="*60)
    print("[COMPARE] 여러 모델 성능 비교")
    print("="*60)
    comparison = predictor.compare_models(
        data['X_train'], data['y_train'],
        data['X_test'], data['y_test']
    )

    print("\n" + "="*60)
    print("[OK] 모델 학습 완료!")
    print("="*60)


if __name__ == "__main__":
    main()