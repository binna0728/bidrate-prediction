import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataProcessor:
    """데이터 전처리 클래스"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = None

    def load_data(self, file_path='강원도및경기일부.xlsx'):
        """데이터 로드"""
        try:
            df = pd.read_excel(file_path)
            print(f"[OK] 데이터 로드 완료: {df.shape[0]}행 x {df.shape[1]}열")
            return df
        except Exception as e:
            print(f"[ERROR] 데이터 로드 실패: {e}")
            return None

    def analyze_columns(self, df):
        """컬럼 분석 및 사정율 컬럼 찾기"""
        print("\n[DATA] 컬럼 분석 중...")

        # 사정율 관련 컬럼 찾기
        rate_columns = []
        for col in df.columns:
            if '사정' in col or '율' in col or '낙찰' in col:
                rate_columns.append(col)
                print(f"  - 발견: {col}")

        # 수치형 컬럼 찾기
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        print(f"\n수치형 컬럼: {len(numeric_cols)}개")

        # 범주형 컬럼 찾기
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        print(f"범주형 컬럼: {len(categorical_cols)}개")

        return {
            'rate_columns': rate_columns,
            'numeric_columns': numeric_cols,
            'categorical_columns': categorical_cols
        }

    def clean_data(self, df):
        """데이터 정제"""
        print("\n[CLEAN] 데이터 정제 중...")

        initial_shape = df.shape

        # 1. 중복 제거
        df = df.drop_duplicates()

        # 2. 결측치 처리
        # 수치형: 평균값으로 대체
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().any():
                # NaN이 아닌 값의 평균 계산
                mean_val = df[col].dropna().mean() if df[col].notna().any() else 0
                df[col].fillna(mean_val, inplace=True)
                print(f"  - {col}: 결측치를 평균값 {mean_val:.2f}로 대체")

        # 범주형: 최빈값으로 대체
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_val, inplace=True)
                print(f"  - {col}: 결측치를 최빈값으로 대체")

        # 3. 이상치 제거 (IQR 방법)
        for col in numeric_cols:
            # NaN이 아닌 값에 대해서만 처리
            if df[col].notna().any():
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                    print(f"  - {col}: {len(outliers)}개 이상치 제거")

        final_shape = df.shape
        print(f"\n[OK] 정제 완료: {initial_shape} → {final_shape}")

        return df

    def prepare_features(self, df, target_column='사정율'):
        """특성 준비"""
        print(f"\n[CONFIG] 특성 준비 중... (타겟: {target_column})")

        # 타겟 컬럼 확인
        if target_column not in df.columns:
            # 사정율 관련 컬럼 자동 찾기
            for col in df.columns:
                if '사정' in col and df[col].dtype in ['float64', 'int64']:
                    target_column = col
                    print(f"  - 타겟 컬럼 자동 선택: {target_column}")
                    break

        if target_column not in df.columns:
            print(f"[ERROR] 타겟 컬럼 '{target_column}'을 찾을 수 없습니다.")
            return None, None, None

        # 타겟과 특성 분리
        y = df[target_column].values
        X = df.drop(columns=[target_column])

        # 날짜 컬럼 제거
        date_cols = X.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            X = X.drop(columns=date_cols)
            print(f"  - 날짜 컬럼 제거: {list(date_cols)}")

        # 범주형 변수 인코딩
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
            print(f"  - {col}: 레이블 인코딩 완료")

        # 특성 이름 저장
        self.feature_columns = X.columns.tolist()
        self.target_column = target_column

        print(f"\n[OK] 특성 준비 완료: {X.shape[1]}개 특성")

        return X, y, self.feature_columns

    def scale_features(self, X_train, X_test=None):
        """특성 스케일링"""
        print("\n[SCALE] 특성 스케일링 중...")

        # 학습 데이터 스케일링
        X_train_scaled = self.scaler.fit_transform(X_train)

        # 테스트 데이터 스케일링
        X_test_scaled = None
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)

        print("[OK] 스케일링 완료")
        return X_train_scaled, X_test_scaled

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """데이터 분할"""
        print(f"\n[DATA] 데이터 분할 중... (테스트 비율: {test_size*100}%)")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        print(f"  - 학습 데이터: {X_train.shape[0]}개")
        print(f"  - 테스트 데이터: {X_test.shape[0]}개")

        return X_train, X_test, y_train, y_test

    def create_feature_dict(self, values):
        """예측을 위한 특성 딕셔너리 생성"""
        if len(values) != len(self.feature_columns):
            raise ValueError(f"입력값 개수({len(values)})가 특성 개수({len(self.feature_columns)})와 일치하지 않습니다.")

        return dict(zip(self.feature_columns, values))

    def process_pipeline(self, file_path='강원도및경기일부.xlsx', target_column='사정율'):
        """전체 처리 파이프라인"""
        print("="*50)
        print("[DATA] 데이터 처리 파이프라인 시작")
        print("="*50)

        # 1. 데이터 로드
        df = self.load_data(file_path)
        if df is None:
            return None

        # 2. 컬럼 분석
        col_info = self.analyze_columns(df)

        # 3. 데이터 정제
        df = self.clean_data(df)

        # 4. 특성 준비
        X, y, feature_columns = self.prepare_features(df, target_column)
        if X is None:
            return None

        # 5. 데이터 분할
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # 6. 스케일링
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)

        # NaN 체크 및 제거
        if np.isnan(X_train_scaled).any():
            print("Warning: NaN found in training data, replacing with 0")
            X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
        if X_test_scaled is not None and np.isnan(X_test_scaled).any():
            print("Warning: NaN found in test data, replacing with 0")
            X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)

        print("\n"+"="*50)
        print("[OK] 데이터 처리 완료!")
        print("="*50)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train': y_train,
            'y_test': y_test,
            'feature_columns': feature_columns,
            'original_data': df,
            'column_info': col_info
        }

if __name__ == "__main__":
    # 테스트 실행
    processor = DataProcessor()
    result = processor.process_pipeline()

    if result:
        print(f"\n최종 결과:")
        print(f"  - 학습 데이터: {result['X_train'].shape}")
        print(f"  - 테스트 데이터: {result['X_test'].shape}")
        print(f"  - 특성 개수: {len(result['feature_columns'])}")
        print(f"  - 특성 목록: {result['feature_columns'][:5]}...")