from typing import List

import pandas as pd


def apply_missing_policy(df: pd.DataFrame, cols: List[str], policy: str) -> pd.DataFrame:
    """
    결측값 처리 정책을 적용한 서브 DataFrame을 반환합니다.

    Parameters
    ----------
    df : pd.DataFrame
        원본 데이터프레임.
    cols : List[str]
        분석에 사용할 열 목록 (그룹 변수 + 분석 변수들).
    policy : str
        결측값 처리 방식. app.py의 라디오 버튼 옵션 문자열과 동일해야 합니다.

    Returns
    -------
    pd.DataFrame
        결측값 정책이 적용된 데이터프레임 (cols만 포함).
    """
    sub = df[cols].copy()

    if policy.startswith("Variable-wise"):
        # 각 분석 함수 내부에서 변수별 dropna를 하기 때문에
        # 여기서는 단순히 서브셋만 반환합니다.
        return sub

    if policy.startswith("Complete-case"):
        # 사용되는 변수들 중 어느 하나라도 NA가 있으면 행을 제거
        return sub.dropna()

    if policy.startswith("Categorical"):
        # 범주형은 'Missing' 카테고리로 포함, 수치는 그대로 둠
        for c in cols:
            if not pd.api.types.is_numeric_dtype(sub[c]):
                sub[c] = sub[c].astype(object).fillna("Missing")
        return sub

    if policy.startswith("Simple imputation"):
        for c in cols:
            if pd.api.types.is_numeric_dtype(sub[c]):
                # 연속형: 중앙값으로 대체
                sub[c] = sub[c].fillna(sub[c].median())
            else:
                # 범주형: 최빈값으로 대체
                if sub[c].mode().empty:
                    sub[c] = sub[c].fillna("Missing")
                else:
                    sub[c] = sub[c].fillna(sub[c].mode().iloc[0])
        return sub

    # 알 수 없는 policy 인 경우, 원본 서브셋 반환
    return sub
