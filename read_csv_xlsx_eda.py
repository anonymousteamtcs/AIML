import os
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split

# ---- Utility helpers ----

def _clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip().replace(' ', '_').lower())
    return df

def _detect_datetime_columns(df: pd.DataFrame, sample_rows: int = 1000) -> List[str]:
    datetime_cols = []
    for col in df.columns:
        if df[col].dtype == 'datetime64[ns]':
            datetime_cols.append(col)
            continue
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            sample = df[col].dropna().astype(str).head(sample_rows)
            if sample.empty:
                continue
            # Try converting; if many convert, treat as datetime
            parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
            success_rate = parsed.notna().mean()
            if success_rate > 0.85:
                datetime_cols.append(col)
    return datetime_cols

def _summarize_objective(df: pd.DataFrame) -> Dict[str, Any]:
    info = {}
    info['rows'] = df.shape[0]
    info['columns'] = df.shape[1]
    info['missing_perc'] = (df.isna().sum() / df.shape[0]).sort_values(ascending=False).to_dict()
    info['dtypes'] = df.dtypes.astype(str).to_dict()
    info['memory_usage_mb'] = df.memory_usage(deep=True).sum() / 1024**2
    return info

def _univariate_stats(df: pd.DataFrame, numeric_only: bool = True) -> pd.DataFrame:
    if numeric_only:
        return df.describe().T
    else:
        return df.describe(include='all').T

def _detect_categorical(df: pd.DataFrame, max_unique_for_cat: int = 50) -> List[str]:
    cats = []
    for col in df.columns:
        if pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            cats.append(col)
        elif pd.api.types.is_integer_dtype(df[col]) or pd.api.types.is_float_dtype(df[col]):
            if df[col].nunique(dropna=True) <= max_unique_for_cat:
                cats.append(col)
    return cats

# ---- Main pipeline function ----

def read_and_preprocess(
    path: str,
    target_column: Optional[str] = None,
    drop_threshold: float = 0.9,
    impute_strategy: str = 'auto',   # options: 'auto', 'mean', 'median', 'mode', 'ffill', 'bfill', 'knn', 'drop'
    knn_k: int = 5,
    scale_method: Optional[str] = 'standard',  # 'standard', 'minmax', or None
    encode_strategy: str = 'auto',  # 'onehot', 'ordinal', 'auto'
    treat_outliers: Optional[str] = None,  # 'iqr', 'zscore', or None
    outlier_threshold: float = 3.0,
    test_size: Optional[float] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Optional[pd.Series], Dict[str, Any]]:
    """
    Read a dataset from .csv or .xlsx and perform comprehensive EDA, cleaning and preprocessing.

    Returns:
      - X_transformed (pd.DataFrame): cleaned and preprocessed features
      - y (pd.Series or None): extracted target if detected or provided
      - report (dict): structured report with EDA and transformation details
    """

    # 1. Load
    if not os.path.exists(path):
        raise FileNotFoundError(f'Path not found: {path}')
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path)
    elif ext in ('.xls', '.xlsx'):
        df = pd.read_excel(path)
    else:
        raise ValueError('Unsupported file extension. Supported: .csv, .xls, .xlsx')

    report: Dict[str, Any] = {}
    report['file_path'] = path
    report['initial_shape'] = df.shape

    # 2. Basic normalization
    df = _clean_column_names(df)
    report['cleaned_col_names'] = list(df.columns)

    # 3. Detect datetimes and parse
    datetime_cols = _detect_datetime_columns(df)
    for col in datetime_cols:
        df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
    report['datetime_columns'] = datetime_cols

    # 4. Drop columns with extremely high missingness
    missing_frac = df.isna().mean()
    cols_to_drop = missing_frac[missing_frac >= drop_threshold].index.tolist()
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
    report['dropped_high_missing_cols'] = cols_to_drop

    # 5. Strip whitespace from object columns and unify empty strings to NaN
    obj_cols = df.select_dtypes(include=['object']).columns
    for c in obj_cols:
        df[c] = df[c].astype(str).str.strip()
        df[c] = df[c].replace({'': np.nan, 'nan': np.nan})
    report['whitespace_stripped'] = list(obj_cols)

    # 6. Remove exact duplicate rows
    dup_rows = df.duplicated().sum()
    if dup_rows:
        df = df.drop_duplicates()
    report['duplicate_rows_removed'] = int(dup_rows)

    # 7. Type coercion: attempt numeric conversion for object columns that look numeric
    for col in df.columns:
        if df[col].dtype == object:
            coerced = pd.to_numeric(df[col].dropna().astype(str).str.replace(',', ''), errors='coerce')
            if coerced.notna().mean() > 0.85:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(',', ''), errors='coerce')

    report['shape_after_basic_cleaning'] = df.shape

    # 8. Summaries
    report['basic_summary'] = _summarize_objective(df)
    report['numeric_descriptives'] = _univariate_stats(df, numeric_only=True).to_dict()
    report['all_descriptives'] = _univariate_stats(df, numeric_only=False).to_dict()

    # 9. Identify potential features vs target
    y = None
    if target_column and target_column in df.columns:
        y = df[target_column]
        X = df.drop(columns=[target_column])
        report['target_source'] = 'provided'
    else:
        # heuristic: if a column named 'target', 'label', 'y', or single low-cardinality non-id
        possible = [c for c in df.columns if c in ('target', 'label', 'y', 'class')]
        if not possible:
            # choose a column with low cardinality and not obviously an id
            candidate = None
            for c in df.columns:
                if df[c].nunique(dropna=True) <= 20 and 'id' not in c:
                    candidate = c
                    break
            if candidate is not None:
                possible = [candidate]
        if possible:
            chosen = possible[0]
            y = df[chosen]
            X = df.drop(columns=[chosen])
            report['target_source'] = f'auto_detected:{chosen}'
        else:
            X = df.copy()
            report['target_source'] = 'none_detected'

    # 10. Detect categorical and numeric columns
    categorical_cols = _detect_categorical(X)
    numeric_cols = [c for c in X.columns if c not in categorical_cols and c not in datetime_cols]
    report['categorical_columns'] = categorical_cols
    report['numeric_columns'] = numeric_cols

    # 11. Missing value handling
    missing_before = X.isna().sum().to_dict()
    if impute_strategy == 'auto':
        # heuristics
        if X.shape[0] < 200 or len(numeric_cols) == 0:
            impute_strategy = 'median'
        else:
            impute_strategy = 'knn' if len(numeric_cols) >= 2 else 'median'

    if impute_strategy == 'drop':
        X = X.dropna()
        if y is not None:
            y = y.loc[X.index]
        impute_info = {'strategy': 'drop'}
    elif impute_strategy in ('mean', 'median', 'mode'):
        X_num = X[numeric_cols]
        if impute_strategy == 'mean':
            fill_vals = X_num.mean()
        elif impute_strategy == 'median':
            fill_vals = X_num.median()
        else:
            fill_vals = X_num.mode().iloc[0] if not X_num.mode().empty else X_num.median()
        X[numeric_cols] = X_num.fillna(fill_vals)
        # categorical: fill with mode
        for c in categorical_cols:
            if X[c].isna().any():
                X[c] = X[c].fillna(X[c].mode().iloc[0] if not X[c].mode().empty else 'missing')
        impute_info = {'strategy': impute_strategy, 'filled_values_numeric_sample': fill_vals.head(10).to_dict()}
    elif impute_strategy in ('ffill', 'bfill'):
        X = X.fillna(method=impute_strategy)
        impute_info = {'strategy': impute_strategy}
    elif impute_strategy == 'knn':
        # Apply KNN imputer on numeric columns; for simplicity, encode categoricals as ordinal temporarily
        knn_imputer = KNNImputer(n_neighbors=max(1, min(knn_k, X.shape[0]-1)))
        X_temp = X.copy()
        # temporary ordinal encode categoricals
        ord_enc = {}
        for c in categorical_cols:
            X_temp[c] = X_temp[c].astype('category').cat.codes.replace({-1: np.nan})
            ord_enc[c] = dict(enumerate(X[c].astype('category').cat.categories))
        # use imputer
        X_temp[numeric_cols + categorical_cols] = knn_imputer.fit_transform(X_temp[numeric_cols + categorical_cols])
        # revert categorical codes to nearest category by rounding codes
        for c in categorical_cols:
            codes = X_temp[c].round().astype(int)
            mapping = ord_enc[c]
            max_idx = max(mapping.keys()) if mapping else -1
            codes = codes.clip(lower=0, upper=max_idx)
            X[c] = codes.map(mapping)
        X[numeric_cols] = X_temp[numeric_cols]
        impute_info = {'strategy': 'knn', 'k': knn_k}
    else:
        impute_info = {'strategy': 'none'}

    report['missing_before'] = missing_before
    report['missing_after'] = X.isna().sum().to_dict()
    report['imputation'] = impute_info

    # 12. Outlier detection and optional treatment
    outlier_info = {}
    if treat_outliers in ('iqr', 'zscore'):
        for col in numeric_cols:
            col_values = X[col].dropna()
            if col_values.empty:
                continue
            if treat_outliers == 'iqr':
                q1 = col_values.quantile(0.25)
                q3 = col_values.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                mask = ~X[col].between(lower, upper)
                outlier_count = mask.sum()
                outlier_info[col] = {'method': 'iqr', 'lower': float(lower), 'upper': float(upper), 'outliers': int(outlier_count)}
                # cap
                X[col] = X[col].clip(lower=lower, upper=upper)
            else:
                mean = col_values.mean()
                std = col_values.std()
                mask = ((X[col] - mean).abs() > outlier_threshold * std)
                outlier_count = mask.sum()
                outlier_info[col] = {'method': 'zscore', 'threshold': outlier_threshold, 'outliers': int(outlier_count)}
                # replace outliers with trimmed value
                X.loc[mask, col] = np.sign(X.loc[mask, col] - mean) * (outlier_threshold * std) + mean
        report['outlier_treatment'] = outlier_info
    else:
        report['outlier_treatment'] = 'none'

    # 13. Encoding categoricals
    encoded_info = {}
    X_enc = X.copy()
    if encode_strategy == 'auto':
        # use one-hot for low-cardinality, ordinal for medium where ordering is likely, drop high-cardinality
        enc_onehot = []
        enc_ordinal = []
        for c in categorical_cols:
            nunique = X[c].nunique(dropna=True)
            if nunique <= 10:
                enc_onehot.append(c)
            elif nunique <= 50:
                enc_ordinal.append(c)
            else:
                # high cardinality -> frequency encode
                freq = X[c].value_counts(normalize=True)
                X_enc[c + '_freq_enc'] = X[c].map(freq).fillna(0)
                X_enc = X_enc.drop(columns=[c])
                encoded_info[c] = {'method': 'frequency', 'unique': nunique}
        if enc_onehot:
            ohe = pd.get_dummies(X_enc[enc_onehot].astype(str), prefix=enc_onehot, dummy_na=False)
            X_enc = pd.concat([X_enc.drop(columns=enc_onehot), ohe], axis=1)
            for c in enc_onehot:
                encoded_info[c] = {'method': 'onehot', 'generated_columns': [col for col in ohe.columns if col.startswith(c + '_')]}
        if enc_ordinal:
            for c in enc_ordinal:
                X_enc[c] = X_enc[c].astype('category').cat.codes
                encoded_info[c] = {'method': 'ordinal', 'unique': int(X[c].nunique(dropna=True))}
    elif encode_strategy == 'onehot':
        X_enc = pd.get_dummies(X_enc, columns=categorical_cols, dummy_na=False)
        encoded_info = {'method': 'onehot'}
    elif encode_strategy == 'ordinal':
        for c in categorical_cols:
            X_enc[c] = X_enc[c].astype('category').cat.codes
        encoded_info = {'method': 'ordinal'}
    else:
        encoded_info = {'method': 'none'}

    report['encoding'] = encoded_info

    # 14. Scaling numeric features
    scaler_info = {}
    if scale_method in ('standard', 'minmax'):
        scaler = StandardScaler() if scale_method == 'standard' else MinMaxScaler()
        if numeric_cols:
            X_enc[numeric_cols] = scaler.fit_transform(X_enc[numeric_cols])
            scaler_info = {'method': scale_method, 'numeric_cols_scaled': numeric_cols}
    else:
        scaler_info = {'method': 'none'}
    report['scaler'] = scaler_info

    # 15. Final sanity checks, target processing
    if y is not None:
        y = y.reset_index(drop=True)
        # if categorical target, keep as-is; if numeric but few unique, convert to category
        if pd.api.types.is_numeric_dtype(y) and y.nunique(dropna=True) <= 20:
            y = y.astype('category')

    X_final = X_enc.reset_index(drop=True)

    # 16. Optionally split
    if test_size is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=test_size, random_state=random_state)
        report['split'] = {'test_size': test_size}
        return (X_train, X_test, y_train, y_test, report)
    else:
        return (X_final, y, report)


# ---- Example usage ----
if __name__ == '__main__':
    # Example (uncomment to run)
    # path = 'data/my_dataset.csv'
    # X, y, r = read_and_preprocess(path, target_column=None, impute_strategy='auto', treat_outliers='iqr', scale_method='standard')
    # print('Report summary:')
    # for k,v in r.items():
    #     print(k, '->', (v if isinstance(v, (str,int,float)) else type(v)))
    pass
