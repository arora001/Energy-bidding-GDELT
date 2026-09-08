import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import logging
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA
from scipy import stats

# Add the parent directory to the path to find the config module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(__file__), 'data_preparation.log'))
    ]
)
logger = logging.getLogger(__name__)

# Now import from config
try:
    from config import PATHS, setup_and_verify
    logger.info("Successfully imported from config module")
except ImportError as e:
    logger.error(f"Failed to import from config: {e}")
    # Define default paths if config import fails
    PATHS = {
        "SPLIT_DATA_DIR": "E:/AI-Energy-Load-New/DATA/split_data",
        "MODELS_DIR": "E:/AI-Energy-Load-New/MODELS",
        "OUTPUT_DIR": "E:/AI-Energy-Load-New/OUTPUT_DIR"
    }
    
    def setup_and_verify():
        """Create necessary directories if they don't exist"""
        for path in PATHS.values():
            os.makedirs(path, exist_ok=True)
        logger.info("Created directories from default paths")

# Update the path configuration section

# Define base directory using the script location
BASE_DIR = Path(__file__).parent.parent.resolve()

# Define data directories relative to the base directory
DATA_DIR = BASE_DIR / "DATA"
SPLIT_DATA_DIR = DATA_DIR / "split_data"
MODEL_DIR = BASE_DIR / "MODELS"
OUTPUT_DIR = BASE_DIR / "OUTPUT_DIR"
MERGED_DATA_DIR = OUTPUT_DIR / "merged_data"

# Create all required directories with proper error handling
def setup_directories():
    """Create all required directories with proper error handling"""
    directories = [
        DATA_DIR,
        SPLIT_DATA_DIR,
        MODEL_DIR,
        OUTPUT_DIR,
        MERGED_DATA_DIR
    ]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created/verified directory: {directory}")
        except PermissionError:
            logger.error(f"Permission denied when creating directory: {directory}")
            # Try creating in user's documents folder as fallback
            user_dir = Path.home() / "Documents" / "AI-Energy-Load"
            logger.info(f"Attempting to create directory in user space: {user_dir}")
            user_dir.mkdir(parents=True, exist_ok=True)
            return str(user_dir)
    return str(BASE_DIR)

# Add these imports at the top of your file, after the existing imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import calendar

def find_merged_data():
    """Find the merged data file across multiple possible locations"""
    if os.path.exists(MERGED_DATA_DIR / "gkg_energy.csv"):
        logger.info(f"Found merged data at primary location: {MERGED_DATA_DIR / 'gkg_energy.csv'}")
        return MERGED_DATA_DIR / "gkg_energy.csv"
        
    alternate_merged_paths = [
        BASE_DIR / "Correlation_and_preprocessing" / "gkg_energy.csv",
        BASE_DIR / "merged_data" / "gkg_energy.csv",
        BASE_DIR / "DATA" / "gkg_energy.csv"
    ]
    
    for alt_path in alternate_merged_paths:
        if os.path.exists(alt_path):
            logger.info(f"Found merged data at alternate location: {alt_path}")
            return alt_path
    
    # Search recursively in the project directory
    logger.info("Searching recursively for merged data file...")
    for root, dirs, files in os.walk(BASE_DIR):
        for file in files:
            if file == "merged_gkg_gkg_energy.csv":
                path = os.path.join(root, file)
                logger.info(f"Found merged data during recursive search: {path}")
                return path
    
    logger.error("Could not find merged data file!")
    return None

def load_data(file_path):
    """Load the merged dataset"""
    logger.info(f"Loading data from {file_path}")
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            logger.error(f"File does not exist: {file_path}")
            return None
            
        # Load the CSV file with proper datetime handling
        df = pd.read_csv(file_path)
        
        # Find datetime column
        datetime_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
        
        if datetime_cols:
            # Use the first datetime column found
            time_col = datetime_cols[0]
            logger.info(f"Using {time_col} as the datetime column")
            df[time_col] = pd.to_datetime(df[time_col])
            df.set_index(time_col, inplace=True)
        else:
            logger.warning("No datetime column found - data will not have time index")
        
        logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} features")
        if isinstance(df.index, pd.DatetimeIndex):
            logger.info(f"Time range: {df.index.min()} to {df.index.max()}")
        
        return df
    except Exception as e:
        logger.error(f"ERROR loading data: {e}", exc_info=True)
        return None

def check_data_quality(df):
    """Perform comprehensive data quality checks"""
    logger.info("Performing data quality checks...")
    
    # 1. Check for missing values
    missing = df.isna().sum()
    missing_cols = missing[missing > 0]
    
    if not missing_cols.empty:
        logger.warning(f"Found {len(missing_cols)} columns with missing values:")
        for col, count in missing_cols.items():
            pct = 100 * count / len(df)
            logger.warning(f"  {col}: {count} missing values ({pct:.2f}%)")
    else:
        logger.info("No missing values found")
    
    # 2. Check for outliers
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    outlier_cols = []
    
    for col in numeric_cols:
        # Use Z-score to detect outliers
        z_scores = np.abs(stats.zscore(df[col].dropna()))
        outliers = (z_scores > 3).sum()
        
        if outliers > 0:
            pct = 100 * outliers / len(df)
            if pct > 1:  # Only flag if more than 1% are outliers
                outlier_cols.append((col, outliers, pct))
    
    if outlier_cols:
        logger.warning(f"Found {len(outlier_cols)} columns with significant outliers:")
        for col, count, pct in outlier_cols:
            logger.warning(f"  {col}: {count} outliers ({pct:.2f}%)")
    else:
        logger.info("No significant outliers found")
    
    # 3. Check for constant or near-constant columns
    const_cols = []
    for col in numeric_cols:
        unique_vals = df[col].nunique()
        if unique_vals <= 1:
            const_cols.append((col, unique_vals, 'constant'))
        elif unique_vals <= 5:
            const_cols.append((col, unique_vals, 'near-constant'))
    
    if const_cols:
        logger.warning(f"Found {len(const_cols)} constant or near-constant columns:")
        for col, vals, status in const_cols:
            logger.warning(f"  {col}: {vals} unique values ({status})")
    
    # 4. Check for duplicated rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.warning(f"Found {duplicates} duplicate rows ({100*duplicates/len(df):.2f}%)")
    else:
        logger.info("No duplicate rows found")
    
    # 5. Check for high correlation between features
    # This can indicate redundant features
    corr_matrix = df[numeric_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    high_corr = [(col1, col2, corr_matrix.loc[col1, col2]) 
                 for col1 in corr_matrix.columns 
                 for col2 in corr_matrix.columns 
                 if corr_matrix.loc[col1, col2] > 0.95 and col1 != col2]
    
    if high_corr:
        logger.warning(f"Found {len(high_corr)} highly correlated feature pairs (r > 0.95):")
        for col1, col2, corr in high_corr[:10]:  # Show top 10
            logger.warning(f"  {col1} & {col2}: {corr:.3f}")
    else:
        logger.info("No extreme feature correlations found")
    
    # Return stats
    return {
        'missing_cols': len(missing_cols),
        'outlier_cols': len(outlier_cols),
        'const_cols': len(const_cols),
        'duplicates': duplicates,
        'high_corr_pairs': len(high_corr)
    }

def handle_missing_values(df):
    """Handle missing values in the dataset with better reporting"""
    logger.info("Handling missing values...")
    
    # Count missing values before
    missing_before = df.isna().sum()
    total_missing = missing_before.sum()
    logger.info(f"Total missing values before handling: {total_missing}")
    
    # For time series, forward-fill is often the best approach
    # for small gaps, followed by backward-fill
    
    # Create a copy to avoid warnings
    df = df.copy()
    
    # First, forward fill (carry last observation forward)
    df = df.ffill()
    
    # Then, backward fill for any remaining NaNs at the beginning
    df = df.bfill()
    
    # For any columns still with NaNs, use column median (more robust than mean)
    still_missing = df.isna().sum()
    
    for col in still_missing[still_missing > 0].index:
        logger.info(f"Using median to fill remaining NaNs in {col}")
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
    
    # Count missing values after
    missing_after = df.isna().sum().sum()
    logger.info(f"Missing values after handling: {missing_after}")
    
    if missing_after > 0:
        logger.warning(f"Could not fill all missing values! {missing_after} still remain.")
    else:
        logger.info("Successfully filled all missing values")
    
    return df

# Define the expected energy column name
EXPECTED_ENERGY_COL = 'Power demand_sum'

def handle_outliers(df, target_col=EXPECTED_ENERGY_COL, method='robust'):
    """
    Handle outliers in the dataset
    
    Parameters:
    - df: DataFrame with the data
    - target_col: Target variable column name (defaults to 'Power demand_sum')
    - method: 'clip', 'robust', or 'none'
    """
    logger.info(f"Handling outliers using '{method}' method...")
    
    # Create a copy
    df = df.copy()
    
    # Verify target column exists
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found. Attempting to find alternative.")
        candidates = [c for c in df.columns if any(x in c.lower() for x in ['power', 'load', 'demand'])]
        if candidates:
            target_col = candidates[0]
            logger.info(f"Using alternative target column: '{target_col}'")
        else:
            logger.error("No suitable target column found for outlier handling. Skipping.")
            return df

    # Get numeric columns, excluding target
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != target_col]
    
    if method == 'none':
        logger.info("Skipping outlier handling (method='none')")
        return df
    
    elif method == 'clip':
        # Clip values to 3 std deviations from mean
        for col in feature_cols:
            # Check if column exists and is numeric before processing
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                mean = df[col].mean()
                std = df[col].std()
                # Avoid issues with zero std deviation
                if std > 0:
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                    
                    # Count outliers before clipping
                    outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
                    if outliers_count > 0:
                        logger.info(f"Clipping {outliers_count} outliers in {col}")
                        
                        # Clip the values
                        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                else:
                    logger.warning(f"Column '{col}' has zero standard deviation. Skipping clipping.")
            else:
                logger.warning(f"Column '{col}' not found or not numeric. Skipping clipping.")
    
    elif method == 'robust':
        # Use robust scaling for all features
        logger.info("Using RobustScaler for all features to handle outliers")
        scaler = RobustScaler()
        
        # Scale all features except target
        if feature_cols: # Ensure there are features to scale
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            
            # Save the scaler
            scaler_path = os.path.join(MODEL_DIR, 'robust_scaler.pkl')
            joblib.dump(scaler, scaler_path)
            logger.info(f"Saved robust scaler to {scaler_path}")
        else:
            logger.warning("No numeric feature columns found to apply RobustScaler.")
    
    else:
        logger.warning(f"Unknown outlier handling method: {method}")
    
    return df

def select_features(X_train, y_train, X_test, feature_selection_method='kbest', k=20):
    """Select most relevant features for the model"""
    logger.info(f"Performing feature selection using {feature_selection_method}...")
    
    if feature_selection_method == 'kbest':
        # Select top k features based on F-regression (linear relationship)
        selector = SelectKBest(score_func=f_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices]
        
        # Log feature scores
        feature_scores = pd.DataFrame({
            'Feature': X_train.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        logger.info("Top 10 feature scores:")
        for idx, row in feature_scores.head(10).iterrows():
            logger.info(f"  {row['Feature']}: {row['Score']:.4f}")
        
    elif feature_selection_method == 'mutual_info':
        # Select based on mutual information (captures non-linear relationships)
        selector = SelectKBest(score_func=mutual_info_regression, k=k)
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # Get selected feature names
        selected_indices = selector.get_support(indices=True)
        selected_features = X_train.columns[selected_indices]
        
        # Log feature scores
        feature_scores = pd.DataFrame({
            'Feature': X_train.columns,
            'Score': selector.scores_
        }).sort_values('Score', ascending=False)
        
        logger.info("Top 10 mutual information scores:")
        for idx, row in feature_scores.head(10).iterrows():
            logger.info(f"  {row['Feature']}: {row['Score']:.4f}")
        
    elif feature_selection_method == 'pca':
        # Use PCA for dimensionality reduction
        pca = PCA(n_components=k)
        X_train_selected = pca.fit_transform(X_train)
        X_test_selected = pca.transform(X_test)
        
        # Create synthetic feature names
        selected_features = [f'PC{i+1}' for i in range(k)]
        
        # Log explained variance
        explained_variance = pca.explained_variance_ratio_
        logger.info(f"Total variance explained: {sum(explained_variance):.2%}")
        logger.info("Top 5 PCA components explained variance:")
        for i, var in enumerate(explained_variance[:5]):
            logger.info(f"  PC{i+1}: {var:.2%}")
            
    else:
        # No feature selection
        logger.info("No feature selection applied")
        X_train_selected = X_train
        X_test_selected = X_test
        selected_features = X_train.columns
    
    # Save feature names
    selection_info = {
        'method': feature_selection_method,
        'features': list(selected_features)
    }
    
    # Save to JSON file
    import json
    selection_path = os.path.join(MODEL_DIR, f'feature_selection_{feature_selection_method}.json')
    with open(selection_path, 'w') as f:
        json.dump(selection_info, f, indent=4)
    
    logger.info(f"Selected {len(selected_features)} features")
    logger.info(f"Feature selection info saved to {selection_path}")
    
    return X_train_selected, X_test_selected, selected_features

def split_time_series_data(df, target_col=EXPECTED_ENERGY_COL, test_fraction=0.2, strategy='month_based', datetime_col=None):
    """
    Split time series data using various strategies
    
    Parameters:
    -----------
    df : DataFrame
    target_col : Target column to predict (defaults to 'Power demand_sum')
    test_fraction : Fraction to use for testing
    strategy : One of ['month_based', 'fully_random', 'seasonal_block']
    datetime_col : Optional; column name to use as datetime for splitting
    """
    logger.info(f"Using '{strategy}' split strategy with {test_fraction:.0%} test fraction")
    
    # Verify target column exists and update if necessary
    if target_col not in df.columns:
        logger.warning(f"Target column '{target_col}' not found. Attempting to find alternative.")
        candidates = [c for c in df.columns if any(x in c.lower() for x in ['power', 'load', 'demand'])]
        if candidates:
            target_col = candidates[0]
            logger.info(f"Using alternative target column: '{target_col}'")
        else:
            # Fallback: Use the last column if no suitable target is found
            target_col = df.columns[-1]
            logger.warning(f"No suitable target column found. Using last column '{target_col}' as target.")
            # Ensure the fallback target is not the datetime column if provided
            if datetime_col is not None and target_col == datetime_col:
                raise ValueError(f"Cannot use the datetime column '{datetime_col}' as the target variable.")
    
    # Sort data chronologically if index is datetime or datetime_col is provided
    if datetime_col is not None and datetime_col in df.columns:
        logger.info(f"Sorting data based on provided datetime column: {datetime_col}")
        df = df.sort_values(datetime_col).copy() # Use copy to avoid SettingWithCopyWarning
    elif isinstance(df.index, pd.DatetimeIndex):
        logger.info("Sorting data based on DatetimeIndex")
        df = df.sort_index().copy() # Use copy
    else:
        logger.warning("No reliable datetime information for sorting.")
        df = df.copy() # Still make a copy
    
    # Create year and month columns for grouping
    # df = df.copy() # Avoid modifying the original dataframe - already copied above
    
    # Use datetime_col if provided and valid
    if datetime_col is not None and datetime_col in df.columns:
        logger.info(f"Using provided datetime column for time grouping: {datetime_col}")
        try:
            dt_series = pd.to_datetime(df[datetime_col])
            df['year'] = dt_series.dt.year
            df['month'] = dt_series.dt.month
            df['season'] = (df['month'] % 12 + 3) // 3 # Seasons: 1=Winter, 2=Spring, 3=Summer, 4=Fall
        except Exception as e:
            logger.error(f"Error processing datetime column '{datetime_col}': {e}. Proceeding without time grouping.")
            df['year'] = 0
            df['month'] = 0
            df['season'] = 0
    
    elif isinstance(df.index, pd.DatetimeIndex):
        logger.info("Using DatetimeIndex for time grouping")
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['season'] = (df['month'] % 12 + 3) // 3
    else:
        # Fallback: Create artificial time grouping ONLY if no other time info is available
        logger.warning("No datetime information found - using artificial time grouping (may be inaccurate)")
        # generate 15‑min intervals to match data granularity
        try:
            random_dates = pd.date_range(start='2020-01-01', periods=len(df), freq='15min') # Use '15min'
            df['temp_datetime'] = random_dates
            df['year'] = df['temp_datetime'].dt.year
            df['month'] = df['temp_datetime'].dt.month
            df['season'] = (df['month'] % 12 + 3) // 3
        except Exception as e:
            logger.error(f"Error creating artificial dates: {e}. Proceeding without time grouping.")
            df['year'] = 0
            df['month'] = 0
            df['season'] = 0
    
    # Get all unique years in the dataset
    years = sorted(df['year'].unique())
    if years == [0]: # Check if only the dummy year exists
        logger.warning("Could not determine valid years for splitting. Using simplified split.")
        # Fallback to simple chronological split if time grouping failed
        test_size = int(len(df) * test_fraction)
        train_df = df.iloc[:-test_size]
        test_df = df.iloc[-test_size:]
        split_info = {'strategy': 'chronological_fallback', 'test_fraction': test_fraction}
        strategy = 'chronological_fallback' # Update strategy name for logging/output
    else:
        logger.info(f"Dataset spans {len(years)} years: {years}")
        
        # STRATEGY 1: Month-based sampling
        if strategy == 'month_based':
            # Create masks for train and test data
            train_mask = np.ones(len(df), dtype=bool)
            
            # For each year, randomly select months for testing
            np.random.seed(42)  # For reproducibility
            test_months_by_year = {}
            
            for year in years:
                # Get available months for this year
                available_months = sorted(df[df['year'] == year]['month'].unique())
                
                # Calculate how many months to select for testing
                n_test_months = max(1, int(len(available_months) * test_fraction))
                
                # Randomly select months for testing
                test_months = sorted(np.random.choice(available_months, size=n_test_months, replace=False))
                # Convert year to standard int for JSON key
                test_months_by_year[int(year)] = test_months 
                
                # Update mask for this year's test months
                year_month_mask = (df['year'] == year) & (df['month'].isin(test_months))
                train_mask[year_month_mask] = False
            
            # Print the selected test months by year
            logger.info("Selected test months by year:")
            for year, months in test_months_by_year.items():
                logger.info(f"  {year}: {[calendar.month_name[m] for m in months]}")
            
            # Split the data based on the masks
            train_df = df[train_mask]
            test_df = df[~train_mask]
            
            # Ensure keys in test_months_by_year are standard Python ints
            test_months_by_year = {int(k): [int(m) for m in v] for k, v in test_months_by_year.items()}
            split_info = {
                'strategy': 'month_based_sampling',
                'test_months_by_year': test_months_by_year
            }
        
        # STRATEGY 2: Fully random sampling (disregards time continuity)
        elif strategy == 'fully_random':
            # Create a random mask with desired ratio
            np.random.seed(42)  # For reproducibility
            test_mask = np.random.rand(len(df)) < test_fraction
            
            train_df = df[~test_mask]
            test_df = df[test_mask]
            
            split_info = {
                'strategy': 'fully_random',
                'test_fraction': test_fraction,
                'test_count': int(test_mask.sum()), # Convert numpy int
                'training_count': int((~test_mask).sum()) # Convert numpy int
            }
            
            logger.info(f"Random split created with {split_info['test_count']} test samples")
        
        # STRATEGY 3: Seasonal blocks (ensure each season is represented)
        elif strategy == 'seasonal_block':
            # Get combinations of year and season
            df['year_season'] = df['year'].astype(str) + "_" + df['season'].astype(str)
            year_seasons = df['year_season'].unique()
            
            # For each year, ensure we sample from each season
            np.random.seed(42)  # For reproducibility
            test_mask = np.zeros(len(df), dtype=bool)
            test_seasons = {}
            
            # Define seasons in order 1→Winter,2→Spring,3→Summer,4→Fall
            season_names = ['Winter', 'Spring', 'Summer', 'Fall']
            
            # Group by year and season
            for year in years:
                current_year_int = int(year)
                test_seasons[current_year_int] = {}
                
                for season in sorted(df[df['year'] == year]['season'].unique()):
                    season_data = df[(df['year'] == year) & (df['season'] == season)]
                    if len(season_data) > 0:
                        n_test_points = int(len(season_data) * test_fraction)
                        # Select consecutive block from middle of the season
                        start_idx = len(season_data) // 2 - n_test_points // 2
                        indices = season_data.index[start_idx:start_idx + n_test_points]
                        test_mask |= df.index.isin(indices)

                        # map 1–4 → index 0–3
                        test_seasons[current_year_int][season_names[season-1]] = n_test_points
            
            train_df = df[~test_mask]
            test_df = df[test_mask]
            
            # Ensure keys and values in test_seasons are standard Python types
            test_seasons_serializable = {}
            for year_key, season_dict in test_seasons.items():
                test_seasons_serializable[int(year_key)] = {k: int(v) for k, v in season_dict.items()}
            
            split_info = {
                'strategy': 'seasonal_block',
                'test_seasons': test_seasons_serializable
            }
        
        else:
            raise ValueError(f"Unknown split strategy: {strategy}. Use 'month_based', 'fully_random', or 'seasonal_block'")
    
    # Remove helper columns before returning
    cols_to_drop = ['year', 'month', 'season', 'year_season', 'temp_datetime']
    train_df = train_df.drop(columns=[col for col in cols_to_drop if col in train_df.columns], errors='ignore')
    test_df = test_df.drop(columns=[col for col in cols_to_drop if col in test_df.columns], errors='ignore')
    
    logger.info(f"Training data: {len(train_df)} rows")
    logger.info(f"Testing data: {len(test_df)} rows")
    
    # Final check for target column existence before splitting X/y
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the training data after splitting.")
    if target_col not in test_df.columns:
        raise ValueError(f"Target column '{target_col}' is missing from the test data after splitting.")
    
    # Create X (features) and y (target) for both sets
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    
    X_test = test_df.drop(target_col, axis=1)
    y_test = test_df[target_col]
    
    return X_train, y_train, X_test, y_test, split_info

def normalize_features(X_train, X_test, y_train, y_test, scaler_type='standard'):
    """Normalize features for better model performance with improved column handling"""
    logger.info(f"Normalizing features using {scaler_type} scaler...")
    
    # First, ensure we only have numeric data
    X_train_numeric = X_train.select_dtypes(include=[np.number]).copy()
    X_test_numeric = X_test.select_dtypes(include=[np.number]).copy()
    
    # Report any non-numeric columns that were dropped
    dropped_columns = set(X_train.columns) - set(X_train_numeric.columns)
    if dropped_columns:
        logger.warning(f"Dropped {len(dropped_columns)} non-numeric columns: {dropped_columns}")
    
    # Initialize scalers based on type
    if scaler_type == 'minmax':
        feature_scaler = MinMaxScaler()
        target_scaler = MinMaxScaler()
    elif scaler_type == 'robust':
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()
    else:  # default to standard
        feature_scaler = StandardScaler()
        target_scaler = MinMaxScaler()  # Usually better to keep target in [0,1]
    
    # Fit and transform the features
    try:
        X_train_scaled = pd.DataFrame(
            feature_scaler.fit_transform(X_train_numeric),
            index=X_train_numeric.index,
            columns=X_train_numeric.columns
        )
        
        X_test_scaled = pd.DataFrame(
            feature_scaler.transform(X_test_numeric),
            index=X_test_numeric.index,
            columns=X_test_numeric.columns
        )
        
        # Reshape y for scaling and apply scaling
        y_train_reshaped = y_train.values.reshape(-1, 1)
        y_train_scaled = target_scaler.fit_transform(y_train_reshaped).flatten()
        
        y_test_reshaped = y_test.values.reshape(-1, 1)
        y_test_scaled = target_scaler.transform(y_test_reshaped).flatten()
        
    except Exception as e:
        logger.error(f"Error during scaling: {e}")
        # Try to identify problematic columns
        problematic_cols = []
        for col in X_train_numeric.columns:
            try:
                # Try to convert to float to see if this column causes issues
                X_train_numeric[col].astype(float)
            except:
                problematic_cols.append(col)
                
        if problematic_cols:
            logger.error(f"Problematic columns: {problematic_cols}")
            # Drop problematic columns and try again
            X_train_numeric = X_train_numeric.drop(columns=problematic_cols)
            X_test_numeric = X_test_numeric.drop(columns=problematic_cols)
            logger.info(f"Retrying scaling after dropping problematic columns")
            
            X_train_scaled = pd.DataFrame(
                feature_scaler.fit_transform(X_train_numeric),
                index=X_train_numeric.index,
                columns=X_train_numeric.columns
            )
            
            X_test_scaled = pd.DataFrame(
                feature_scaler.transform(X_test_numeric),
                index=X_test_numeric.index,
                columns=X_test_numeric.columns
            )
            
            y_train_reshaped = y_train.values.reshape(-1, 1)
            y_train_scaled = target_scaler.fit_transform(y_train_reshaped).flatten()
            
            y_test_reshaped = y_test.values.reshape(-1, 1)
            y_test_scaled = target_scaler.transform(y_test_reshaped).flatten()
    
    # Save the scalers for later use
    scaler_dir = os.path.join(MODEL_DIR, 'scalers')
    os.makedirs(scaler_dir, exist_ok=True)
    
    joblib.dump(feature_scaler, os.path.join(scaler_dir, f'feature_scaler_{scaler_type}.pkl'))
    joblib.dump(target_scaler, os.path.join(scaler_dir, f'target_scaler_{scaler_type}.pkl'))
    
    logger.info(f"Scalers saved to {scaler_dir}")
    
    # Check for proper scaling
    logger.info("Verifying scaled data ranges:")
    
    # Features should be properly scaled
    X_train_min = X_train_scaled.min().mean()
    X_train_max = X_train_scaled.max().mean()
    logger.info(f"X_train scaled range: [{X_train_min:.3f}, {X_train_max:.3f}] (avg across features)")
    
    # Target should be properly scaled
    y_train_min = np.min(y_train_scaled)
    y_train_max = np.max(y_train_scaled)
    logger.info(f"y_train scaled range: [{y_train_min:.3f}, {y_train_max:.3f}]")
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler, X_train_numeric.columns

def save_split_data(X_train, y_train, X_test, y_test, train_idx, test_idx, split_info, strategy_name="default"):
    """Save the split datasets in separate train/test directories"""
    logger.info(f"Saving split data for {strategy_name} strategy")
    
    # Create a timestamp for the filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create train and test subdirectories
    train_dir = os.path.join(SPLIT_DATA_DIR, "train_data")
    test_dir = os.path.join(SPLIT_DATA_DIR, "test_data")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Save training data
    train_df = pd.DataFrame(X_train, index=train_idx)
    train_df['target'] = y_train
    train_path = os.path.join(train_dir, f'train_data_{strategy_name}_{timestamp}.csv')
    train_df.to_csv(train_path)
    logger.info(f"Training data saved to {train_path}")
    
    # Save testing data
    test_df = pd.DataFrame(X_test, index=test_idx)
    test_df['target'] = y_test
    test_path = os.path.join(test_dir, f'test_data_{strategy_name}_{timestamp}.csv')
    test_df.to_csv(test_path)
    logger.info(f"Testing data saved to {test_path}")
    
    # Save metadata about the split
    meta_dir = os.path.join(SPLIT_DATA_DIR, "metadata")
    os.makedirs(meta_dir, exist_ok=True)
    
    with open(os.path.join(meta_dir, f'split_info_{strategy_name}_{timestamp}.txt'), 'w') as f:
        f.write(f"Data split performed on: {datetime.now()}\n")
        f.write(f"Split strategy: {strategy_name}\n\n")
        
        # Write detailed strategy info
        if isinstance(split_info, dict):
            for key, value in split_info.items():
                if key == 'test_months_by_year' and isinstance(value, dict):
                    f.write("\nTest months by year:\n")
                    for year, months in value.items():
                        f.write(f"  {year}: {[calendar.month_name[m] for m in months]}\n")
                elif key == 'test_seasons' and isinstance(value, dict):
                    f.write("\nTest seasons by year:\n")
                    for year, seasons in value.items():
                        f.write(f"  {year}: {seasons}\n")
                else:
                    f.write(f"{key}: {value}\n")
        else:
            f.write(f"Split info: {split_info}\n")
            
        f.write(f"\nTraining data: {len(train_df)} rows\n")
        f.write(f"Testing data: {len(test_df)} rows\n")
    
    # Also save split visualization
    create_split_visualization(train_idx, test_idx, y_train, y_test, split_info, meta_dir, timestamp, strategy_name)
    
    return train_path, test_path

def create_split_visualization(train_idx, test_idx, y_train, y_test, split_info, output_dir, timestamp, strategy_name):
    """Create and save visualizations of the data split"""
    
    # Create power demand plot
    plt.figure(figsize=(14, 7))
    
    # Plot all data points as small dots
    all_idx = train_idx.union(test_idx)
    all_y = pd.Series(np.nan, index=all_idx)
    all_y.loc[train_idx] = y_train
    all_y.loc[test_idx] = y_test
    
    plt.plot(all_idx, all_y, 'k.', alpha=0.1, markersize=1)
    
    # Plot training data
    plt.plot(train_idx, y_train, 'b.', alpha=0.5, label='Training Data')
    
    # Plot test data with higher visibility
    plt.plot(test_idx, y_test, 'r.', alpha=0.7, label='Testing Data')
    
    plt.title(f'Energy Load Training/Testing Split ({strategy_name})')
    plt.xlabel('Date')
    plt.ylabel('Power Demand')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the visualization
    viz_path = os.path.join(output_dir, f'split_visualization_{strategy_name}_{timestamp}.png')
    plt.savefig(viz_path, dpi=300)
    logger.info(f"Visualization saved to {viz_path}")
    
    # Also create a monthly distribution visualization
    plt.figure(figsize=(12, 6))
    
    train_counts = train_idx.month.value_counts().sort_index()
    test_counts = test_idx.month.value_counts().sort_index()
    
    months = range(1, 13)
    month_names = [calendar.month_abbr[m] for m in months]
    
    train_values = [train_counts.get(m, 0) for m in months]
    test_values = [test_counts.get(m, 0) for m in months]
    
    bar_width = 0.35
    r1 = np.arange(len(months))
    r2 = [x + bar_width for x in r1]
    
    plt.bar(r1, train_values, width=bar_width, label='Training', color='blue', alpha=0.7)
    plt.bar(r2, test_values, width=bar_width, label='Testing', color='red', alpha=0.7)
    
    plt.xlabel('Month')
    plt.ylabel('Number of Samples')
    plt.title(f'Monthly Distribution of Training and Test Data ({strategy_name})')
    plt.xticks([r + bar_width/2 for r in range(len(months))], month_names)
    plt.legend()
    plt.tight_layout()
    
    # Save the monthly distribution visualization
    monthly_viz_path = os.path.join(output_dir, f'monthly_distribution_{strategy_name}_{timestamp}.png')
    plt.savefig(monthly_viz_path, dpi=300)
    logger.info(f"Monthly distribution visualization saved to {monthly_viz_path}")

def create_sequences(X, y, seq_length=24):
    """
    Create sequences for LSTM input
    seq_length: number of time steps in each sequence (24 = 6 hours at 15-min intervals)
    """
    X_seq = []
    y_seq = []
    
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i+seq_length])
        # Predict the next time step
        y_seq.append(y[i+seq_length])
        
    return np.array(X_seq), np.array(y_seq)

def get_walk_forward_splits(df, initial_train_pct=0.8, test_pct=0.02):
    """
    Generator that yields train/test splits using walk-forward logic.
    """
    total_len = len(df)
    initial_train_size = int(total_len * initial_train_pct)
    test_size = int(total_len * test_pct)

    end = initial_train_size
    while end + test_size <= total_len:
        train = df.iloc[:end]
        test = df.iloc[end:end + test_size]
        yield train, test
        end += test_size
def save_cleaned_data(df, filename="gkg_energy.csv"):
    from config import PATHS
    import os

    merged_dir = os.path.join(PATHS["BASE_DIR"], "merged_data")
    os.makedirs(merged_dir, exist_ok=True)
    save_path = os.path.join(merged_dir, filename)

    df.to_csv(save_path)
    print(f"✅ Cleaned dataset saved to {save_path}")

def main():
    """Main execution function"""
    logger.info("=== Energy Load Prediction Data Preparation ===")
    
    # Ensure directories exist
    setup_directories()
    
    # Create additional directories for test/train data
    train_dir = os.path.join(SPLIT_DATA_DIR, "train_data")
    test_dir = os.path.join(SPLIT_DATA_DIR, "test_data")
    meta_dir = os.path.join(SPLIT_DATA_DIR, "metadata")
    
    for directory in [train_dir, test_dir, meta_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Load merged data
    merged_data_path = find_merged_data()
    if not merged_data_path:
        logger.error("Failed to find merged data. Exiting.")
        return
    
    merged_df = load_data(merged_data_path)
    if merged_df is None:
        logger.error("Failed to load data. Exiting.")
        return
    
    # Check data quality
    data_quality_stats = check_data_quality(merged_df)
    
    # Clean the data
    merged_df = handle_missing_values(merged_df)
    # Pass the correct target column name to handle_outliers
    merged_df = handle_outliers(merged_df, target_col=EXPECTED_ENERGY_COL)
    
    # Try all three split strategies
    strategies = ['month_based', 'fully_random', 'seasonal_block']
    
    for strategy in strategies:
        logger.info(f"\n{'='*50}")
        logger.info(f"Processing {strategy.upper()} split strategy")
        logger.info(f"{'='*50}")
        
        # Split the data with current strategy, passing the correct target column
        X_train, y_train, X_test, y_test, split_info = split_time_series_data(
            merged_df,
            target_col=EXPECTED_ENERGY_COL, # Pass the correct target column
            test_fraction=0.2,
            strategy=strategy
        )
        
        # Save original indices before normalization
        train_idx = X_train.index
        test_idx = X_test.index
        
        # Normalize the data
        X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, feature_scaler, target_scaler, feature_names = normalize_features(
            X_train, X_test, y_train, y_test
        )
        
        # Select features
        # Ensure y_train_scaled is passed correctly (it's already scaled)
        X_train_selected, X_test_selected, selected_features = select_features(
            X_train_scaled, y_train_scaled, X_test_scaled # Pass scaled y_train
        )
        
        # Save the split data with strategy in filename
        train_path, test_path = save_split_data(
            X_train_selected, y_train_scaled, 
            X_test_selected, y_test_scaled,
            train_idx, test_idx, split_info,
            strategy_name=strategy
        )
        
        logger.info(f"\n{strategy} data preparation completed!")
        logger.info(f"Training set shape: {X_train_selected.shape}")
        logger.info(f"Testing set shape: {X_test_selected.shape}")
        # logger.info(f"Feature count: {X_train_selected.shape[1]} features") # Shape already includes feature count
    
    logger.info("\nAll split strategies processed successfully!")

if __name__ == "__main__":
    import logging
    from config import PATHS

    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

    logger.info("=== Running Full Data Preparation ===")

    # --- Step 1: Load Raw Aggregated News + Weather Data ---
    news_path = os.path.join(PATHS['AGGREGATED_DIR'], 'aggregated_gkg_15min.csv')
    weather_path = os.path.join(PATHS['AGGREGATED_DIR'], 'weather_energy_15min.csv')

    if not os.path.exists(news_path):
        logger.error(f"Missing news data: {news_path}")
        exit(1)
    if not os.path.exists(weather_path):
        logger.error(f"Missing weather data: {weather_path}")
        exit(1)

    news_df = pd.read_csv(news_path, index_col=0, parse_dates=True)
    weather_df = pd.read_csv(weather_path, index_col=0, parse_dates=True)

    # --- Step 2: Merge on datetime index ---
    df = news_df.join(weather_df, how="inner")
    logger.info(f"Merged shape: {df.shape}")

    # --- Step 3: Clean + Sort ---
    df = handle_missing_values(df)
    df = handle_outliers(df, target_col="Power demand_sum")
    df = df.sort_index()

    # --- Step 4: Save Final Cleaned Dataset ---
    from Model_pedictor.preparation import save_cleaned_data
    save_cleaned_data(df)

    merged_data_output_path = os.path.join(PATHS['BASE_DIR'], 'merged_data', 'gkg_energy.csv')
    os.makedirs(os.path.dirname(merged_data_output_path), exist_ok=True)
    df.to_csv(merged_data_output_path)
    logger.info(f"✅ Final merged file saved at: {merged_data_output_path}")
    logger.info("✅ Preparation complete.")