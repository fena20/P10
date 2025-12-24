"""
Feature Builder for Heating Demand Modeling

Implements feature engineering as specified in Section 5 and 7.
All transformations that could leak information must be fit inside CV inner loop.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from src.utils.helpers import logger, Timer, validate_no_leakage


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Feature builder for RECS 2020 heating demand modeling.
    
    Implements Section 5: Features and the "2020/COVID problem"
    and Section 7: Leakage-proof preprocessing
    
    Key principles:
    - All transformations fitted inside CV inner loop only
    - Explicit handling of COVID controls
    - Leakage blacklist enforcement
    """
    
    # Feature categories
    CLIMATE_FEATURES = [
        'HDD65', 'CDD65', 'HDD30YR_PUB', 'CDD30YR_PUB'
    ]
    
    BUILDING_FEATURES = [
        'TOTSQFT_EN', 'TOTHSQFT', 'TOTCSQFT',
        'TYPEHUQ', 'YEARMADERANGE', 'STORIES',
        'BEDROOMS', 'TOTROOMS',
        'CELLAR', 'CRAWL', 'CONCRETE', 'ATTIC', 'ATTICFIN',
        'WALLTYPE', 'ROOFTYPE', 'WINDOWS', 'TYPEGLASS', 'WINFRAME',
        'ADQINSUL', 'DRAFTY', 'HIGHCEIL', 'TREESHAD'
    ]
    
    EQUIPMENT_FEATURES = [
        'EQUIPM', 'FUELHEAT', 'EQUIPAGE',
        'EQUIPAUX', 'FUELAUX', 'USEEQUIPAUX',
        'HEATCNTL', 'TEMPHOME',
        'BASEHEAT', 'ATTCHEAT', 'GARGHEAT'
    ]
    
    OCCUPANCY_FEATURES = [
        'NHSLDMEM', 'NUMCHILD', 'NUMADULT1', 'NUMADULT2',
        'HHAGE', 'EMPLOYHH', 'KOWNRENT'
    ]
    
    COVID_DIRECT_FEATURES = [
        'TELLWORK',  # Telework indicator
        'ATHOME'     # Someone at home during day
    ]
    
    GEOGRAPHIC_FEATURES = [
        'DIVISION', 'BA_climate', 'IECC_climate_code', 'UATYP10'
    ]
    
    # Leakage blacklist (Section 7)
    LEAKAGE_BLACKLIST = [
        # Direct cost/expenditure variables linked to heating use
        'DOLLAREL', 'DOLLARNG', 'DOLLARLP', 'DOLLARFO',
        'TOTALDOL', 'TOTALDOLSPH',
        'DOLELSPH', 'DOLNGSPH', 'DOLLPSPH', 'DOLFOSPH',
        # BTU components that sum to target
        'BTUELSPH', 'BTUNGSPH', 'BTULPSPH', 'BTUFOSPH',
        # kWh equivalents
        'KWHSPH', 'BTUEL',
        # Other end-use disaggregations
        'KWH', 'BTUNG', 'BTULP', 'BTUFO', 'TOTALBTU'
    ]
    
    # Categorical features for encoding
    CATEGORICAL_FEATURES = [
        'TYPEHUQ', 'YEARMADERANGE', 'DIVISION',
        'WALLTYPE', 'ROOFTYPE', 'TYPEGLASS', 'WINFRAME',
        'BA_climate', 'IECC_climate_code', 'UATYP10',
        'EQUIPM', 'FUELHEAT', 'FUELAUX',
        'KOWNRENT', 'HEATCNTL',
        # Optional: include when tech_group is present in X and requested via get_feature_columns(...)
        'tech_group'
    ]
    
    # Ordinal features (can be treated as numeric)
    ORDINAL_FEATURES = [
        'ADQINSUL', 'DRAFTY', 'WINDOWS', 'STORIES',
        'EQUIPAGE', 'TEMPHOME'
    ]
    
    def __init__(self,
                 covid_control_mode: str = 'direct',
                 include_interactions: bool = True,
                 scale_continuous: bool = False,
                 encode_categorical: str = 'onehot',
                 handle_missing: str = 'indicator'):
        """
        Initialize feature builder.
        
        Parameters
        ----------
        covid_control_mode : str
            How to handle COVID controls:
            - 'direct': Include COVID indicators (TELLWORK, ATHOME)
            - 'proxy': Use proxy occupancy variables only
            - 'none': Exclude all COVID-related controls
        include_interactions : bool
            Whether to include interaction features
        scale_continuous : bool
            Whether to scale continuous features
        encode_categorical : str
            Categorical encoding: 'onehot', 'ordinal', or 'target'
        handle_missing : str
            Missing value handling: 'indicator', 'impute', or 'drop'
        """
        self.covid_control_mode = covid_control_mode
        self.include_interactions = include_interactions
        self.scale_continuous = scale_continuous
        self.encode_categorical = encode_categorical
        self.handle_missing = handle_missing
        
        # Fitted transformers (set during fit)
        self.scaler_ = None
        self.encoder_ = None
        self.imputer_ = None
        self.feature_names_ = None
        self.continuous_features_ = None
        self.categorical_features_ = None
        
    def get_feature_columns(self, 
                            include_tech_group: bool = False) -> List[str]:
        """
        Get list of feature columns based on configuration.
        
        Parameters
        ----------
        include_tech_group : bool
            Whether to include tech_group as a feature
            
        Returns
        -------
        list
            List of feature column names
        """
        features = []
        
        # Climate features
        features.extend(self.CLIMATE_FEATURES)
        
        # Building features
        features.extend(self.BUILDING_FEATURES)
        
        # Equipment features
        features.extend(self.EQUIPMENT_FEATURES)
        
        # Occupancy features
        features.extend(self.OCCUPANCY_FEATURES)
        
        # Geographic features
        features.extend(self.GEOGRAPHIC_FEATURES)
        
        # COVID controls based on mode
        if self.covid_control_mode == 'direct':
            features.extend(self.COVID_DIRECT_FEATURES)
        elif self.covid_control_mode == 'proxy':
            # Use occupancy proxies already in OCCUPANCY_FEATURES
            pass
        # 'none' mode: don't add COVID features
        
        # Technology group
        if include_tech_group:
            features.append('tech_group')
        
        # Remove duplicates while preserving order
        seen = set()
        unique_features = []
        for f in features:
            if f not in seen:
                seen.add(f)
                unique_features.append(f)
        
        # Validate no leakage
        leakage_found = validate_no_leakage(unique_features, self.LEAKAGE_BLACKLIST)
        if leakage_found:
            raise ValueError(f"Leakage features found in feature list: {leakage_found}")
        
        return unique_features
    
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'FeatureBuilder':
        """
        Fit the feature builder on training data.
        
        IMPORTANT: This must only be called on training data within CV inner loop
        to prevent leakage.
        
        Parameters
        ----------
        X : DataFrame
            Training features
        y : Series, optional
            Training target (for target encoding if used)
            
        Returns
        -------
        self
        """
        logger.debug("Fitting feature builder")
        
        # Identify continuous and categorical features present in data
        # Respect whether the caller included tech_group as a feature.
        feature_cols = self.get_feature_columns(include_tech_group=('tech_group' in X.columns))
        available_features = [f for f in feature_cols if f in X.columns]
        
        self.continuous_features_ = [
            f for f in available_features 
            if f not in self.CATEGORICAL_FEATURES and f not in ['tech_group']
        ]
        self.categorical_features_ = [
            f for f in available_features 
            if f in self.CATEGORICAL_FEATURES
        ]
        
        # Fit imputer for continuous features
        if self.handle_missing in ['indicator', 'impute']:
            if self.continuous_features_:
                self.imputer_ = SimpleImputer(strategy='median')
                self.imputer_.fit(X[self.continuous_features_])
        
        # Fit scaler for continuous features
        if self.scale_continuous and self.continuous_features_:
            self.scaler_ = StandardScaler()
            X_cont = X[self.continuous_features_].copy()
            if self.imputer_ is not None:
                X_cont = pd.DataFrame(
                    self.imputer_.transform(X_cont),
                    columns=self.continuous_features_
                )
            self.scaler_.fit(X_cont)
        
        # Fit encoder for categorical features
        if self.encode_categorical == 'onehot' and self.categorical_features_:
            self.encoder_ = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                drop='first'  # Avoid dummy variable trap
            )
            X_cat = X[self.categorical_features_].fillna(-999).astype(str)
            self.encoder_.fit(X_cat)
        
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features.
        
        Parameters
        ----------
        X : DataFrame
            Features to transform
            
        Returns
        -------
        DataFrame
            Transformed features
        """
        X_transformed = X.copy()
        result_dfs = []
        
        # Transform continuous features
        if self.continuous_features_:
            X_cont = X_transformed[self.continuous_features_].copy()
            
            # Impute missing values
            if self.imputer_ is not None:
                X_cont = pd.DataFrame(
                    self.imputer_.transform(X_cont),
                    columns=self.continuous_features_,
                    index=X.index
                )
            
            # Add missing indicators if requested
            if self.handle_missing == 'indicator':
                for col in self.continuous_features_:
                    missing_col = f'{col}_missing'
                    X_cont[missing_col] = X[col].isna().astype(int)
            
            # Scale
            if self.scaler_ is not None:
                X_cont_scaled = pd.DataFrame(
                    self.scaler_.transform(X_cont[self.continuous_features_]),
                    columns=self.continuous_features_,
                    index=X.index
                )
                # Replace continuous columns with scaled versions
                for col in self.continuous_features_:
                    X_cont[col] = X_cont_scaled[col]
            
            result_dfs.append(X_cont)
        
        # Transform categorical features
        if self.categorical_features_:
            X_cat = X_transformed[self.categorical_features_].fillna(-999).astype(str)
            
            if self.encode_categorical == 'onehot' and self.encoder_ is not None:
                X_encoded = pd.DataFrame(
                    self.encoder_.transform(X_cat),
                    columns=self.encoder_.get_feature_names_out(self.categorical_features_),
                    index=X.index
                )
                result_dfs.append(X_encoded)
            elif self.encode_categorical == 'ordinal':
                # Simple label encoding
                X_ordinal = X_cat.copy()
                for col in self.categorical_features_:
                    X_ordinal[col] = pd.factorize(X_cat[col])[0]
                result_dfs.append(X_ordinal)
            else:
                result_dfs.append(X_cat)
        
        # tech_group is treated as a categorical feature and is encoded via the same fold-fitted
        # encoder as other categoricals (anti-leakage). Keep this fallback only if tech_group was
        # intentionally excluded from the categorical encoder.
        if 'tech_group' in X_transformed.columns and (self.categorical_features_ is None or 'tech_group' not in self.categorical_features_):
            tech_dummies = pd.get_dummies(
                X_transformed['tech_group'],
                prefix='tech',
                drop_first=True
            )
            result_dfs.append(tech_dummies)
        
        # Combine all features
        X_final = pd.concat(result_dfs, axis=1)
        
        # Add interaction features
        if self.include_interactions:
            X_final = self._add_interactions(X_final, X_transformed)
        
        self.feature_names_ = list(X_final.columns)
        
        return X_final
    
    def fit_transform(self, X: pd.DataFrame, 
                      y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def _add_interactions(self, X_transformed: pd.DataFrame,
                          X_original: pd.DataFrame) -> pd.DataFrame:
        """
        Add interaction features.

        Physics-motivated interactions to address (i) scale compression / heavy tail
        and (ii) systematic subgroup bias.

        Implemented (subset):
        - HDD × Area
        - HDD × Vintage (YEARMADERANGE: ordinal or one-hot)
        - HDD × Tech (if tech dummies present)
        - Area × HousingType (TYPEHUQ one-hot)
        - HDD × (ADQINSUL, DRAFTY) when available
        """
        X = X_transformed.copy()

        # Helper: pull series from transformed if present; otherwise from original
        def _get_series(col: str) -> Optional[pd.Series]:
            if col in X.columns:
                return X[col]
            if col in X_original.columns:
                return X_original[col]
            return None

        hdd = _get_series('HDD65')
        sqft = _get_series('TOTSQFT_EN')
        
        # HDD × Area interaction
        if hdd is not None and sqft is not None:
            X['HDD_x_SQFT'] = hdd * sqft
        
        # HDD squared (nonlinear climate effect)
        if hdd is not None:
            X['HDD65_sq'] = hdd ** 2
        
        # SQFT squared
        if sqft is not None:
            X['SQFT_sq'] = sqft ** 2

        # HDD × insulation / draftiness (ordinal-ish, treated numeric here)
        if hdd is not None:
            if 'ADQINSUL' in X.columns:
                X['HDD_x_ADQINSUL'] = hdd * X['ADQINSUL']
            if 'DRAFTY' in X.columns:
                X['HDD_x_DRAFTY'] = hdd * X['DRAFTY']

        # HDD × Vintage
        # If one-hot encoded: interact with each dummy; if ordinal: interact with numeric column
        if hdd is not None:
            vintage_cols = [c for c in X.columns if c.startswith('YEARMADERANGE_')]
            if vintage_cols:
                for c in vintage_cols:
                    X[f'HDD_x_{c}'] = hdd * X[c]
            elif 'YEARMADERANGE' in X.columns:
                X['HDD_x_YEARMADERANGE'] = hdd * X['YEARMADERANGE']

        # HDD × Technology (only meaningful for monolithic models that include tech dummies)
        if hdd is not None:
            tech_cols = [c for c in X.columns if c.startswith('tech_')]
            for c in tech_cols:
                X[f'HDD_x_{c}'] = hdd * X[c]

        # Area × HousingType (helps Apartment 5+ group and size-response heterogeneity)
        if sqft is not None:
            ht_cols = [c for c in X.columns if c.startswith('TYPEHUQ_')]
            for c in ht_cols:
                X[f'SQFT_x_{c}'] = sqft * X[c]

            # Optional: HDD×HousingType to allow different climate sensitivity by housing type
            if hdd is not None:
                for c in ht_cols:
                    X[f'HDD_x_{c}'] = hdd * X[c]
        
        return X
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """
        Get feature names grouped by category for interpretation.
        
        Returns
        -------
        dict
            Dictionary mapping category names to feature lists
        """
        if self.feature_names_ is None:
            raise ValueError("Feature builder not fitted. Call fit_transform first.")
        
        groups = {
            'climate': [],
            'building': [],
            'equipment': [],
            'occupancy': [],
            'geographic': [],
            'covid': [],
            'interactions': [],
            'other': []
        }
        
        for feat in self.feature_names_:
            feat_base = feat.split('_')[0] if '_' in feat else feat
            
            if any(f in feat for f in ['HDD', 'CDD']):
                groups['climate'].append(feat)
            elif any(f in feat for f in self.BUILDING_FEATURES):
                groups['building'].append(feat)
            elif any(f in feat for f in self.EQUIPMENT_FEATURES):
                groups['equipment'].append(feat)
            elif any(f in feat for f in self.OCCUPANCY_FEATURES):
                groups['occupancy'].append(feat)
            elif any(f in feat for f in self.GEOGRAPHIC_FEATURES):
                groups['geographic'].append(feat)
            elif any(f in feat for f in self.COVID_DIRECT_FEATURES):
                groups['covid'].append(feat)
            elif '_x_' in feat or '_sq' in feat:
                groups['interactions'].append(feat)
            else:
                groups['other'].append(feat)
        
        return groups


class LeakageFreeTransformer(BaseEstimator, TransformerMixin):
    """
    Wrapper to ensure transformations are leakage-free in CV.
    
    This wrapper enforces that:
    1. fit() is only called on training data
    2. transform() uses only fitted parameters
    3. No leakage features are included
    """
    
    def __init__(self, feature_builder: FeatureBuilder):
        self.feature_builder = feature_builder
        self.is_fitted_ = False
        
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'LeakageFreeTransformer':
        """Fit on training data only."""
        self.feature_builder.fit(X, y)
        self.is_fitted_ = True
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform using fitted parameters."""
        if not self.is_fitted_:
            raise ValueError("Transformer not fitted. Call fit() first.")
        return self.feature_builder.transform(X)
    
    def fit_transform(self, X: pd.DataFrame, 
                      y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform."""
        return self.fit(X, y).transform(X)


def create_feature_matrix(df: pd.DataFrame,
                          target_col: str = 'TOTALBTUSPH',
                          weight_col: str = 'NWEIGHT',
                          covid_mode: str = 'direct',
                          include_tech_group: bool = True) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Convenience function to create feature matrix for modeling.
    
    Parameters
    ----------
    df : DataFrame
        Preprocessed RECS data
    target_col : str
        Target column name
    weight_col : str
        Weight column name
    covid_mode : str
        COVID control mode
    include_tech_group : bool
        Whether to include technology group
        
    Returns
    -------
    X : DataFrame
        Feature matrix
    y : Series
        Target variable
    weights : Series
        Sample weights
    """
    builder = FeatureBuilder(covid_control_mode=covid_mode)
    feature_cols = builder.get_feature_columns(include_tech_group=include_tech_group)
    
    # Filter to available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[available_cols].copy()
    y = df[target_col].copy()
    weights = df[weight_col].copy()
    
    return X, y, weights
