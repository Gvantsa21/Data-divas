# DATA EXPLORATION - SPOTIFY TRACK POPULARITY DATASET
 
# DATASET OVERVIEW
# This is a Spotify tracks dataset containing information about 114,000 songs
# The dataset has 21 columns (features) describing various aspects of each track
# Memory usage: approximately 17.5 MB
 
 
# COLUMN STRUCTURE (21 total columns)
 
# IDENTIFIER COLUMNS (2):
# - id: Numeric identifier for each track (int64, 114000 entries)
# - track_id: Unique Spotify track ID (object/string, 114000 entries)
 
# TRACK METADATA (3):
# - artists: Artist name(s) (object/string, 113999 entries - 1 missing value)
# - album_name: Name of the album (object/string, 113999 entries - 1 missing value)
# - track_name: Name of the song (object/string, 113999 entries - 1 missing value)
 
# TARGET VARIABLE (1):
# - popularity: Popularity score from 0-100 (int64, 114000 entries, no missing values)
#   This is what we want to predict
 
# TRACK CHARACTERISTICS (4):
# - duration_ms: Track length in milliseconds (int64, 114000 entries)
# - explicit: Whether track contains explicit content (bool, 114000 entries)
# - key: Musical key (0-11 representing C through B) (int64, 114000 entries)
# - mode: Major (1) or Minor (0) mode (int64, 114000 entries)
# - time_signature: Time signature (int64, 114000 entries)
 
# AUDIO FEATURES (9):
# All audio features are normalized between 0 and 1 (float64 type)
# - danceability: How suitable for dancing (float64, 114000 entries)
# - energy: Intensity and activity measure (float64, 114000 entries)
# - loudness: Overall loudness in dB (float64, 114000 entries)
# - speechiness: Presence of spoken words (float64, 114000 entries)
# - acousticness: Confidence the track is acoustic (float64, 114000 entries)
# - instrumentalness: Predicts if track has no vocals (float64, 114000 entries)
# - liveness: Presence of audience in recording (float64, 114000 entries)
# - valence: Musical positiveness/happiness (float64, 114000 entries)
# - tempo: Overall estimated tempo in BPM (float64, 114000 entries)
 
# GENRE (1):
# - track_genre: Genre classification (object/string, 114000 entries)
 
 
# DATA TYPES SUMMARY
# - bool: 1 column (explicit)
# - float64: 9 columns (all audio features)
# - int64: 6 columns (id, popularity, duration_ms, key, mode, time_signature)
# - object: 5 columns (track_id, artists, album_name, track_name, track_genre)
 
 
# MISSING VALUES
# Total missing values: 3 (only 0.0026% of entire dataset)
# - artists: 1 missing value
# - album_name: 1 missing value
# - track_name: 1 missing value
# All other columns have complete data (no missing values)
 
 
# DATASET QUALITY
# - Very clean dataset with minimal missing data
# - No missing values in any numerical features
# - No missing values in target variable (popularity)
# - Only 3 missing values total in categorical text fields
# - All 114,000 rows are valid entries
 
 
# INITIAL OBSERVATIONS
# - Large dataset: 114,000 tracks provide substantial data for analysis
# - Comprehensive features: 21 different attributes per track
# - Complete audio analysis: 9 Spotify-calculated audio features
# - Minimal data cleaning needed: Only 3 missing values to handle
# - Target variable complete: All tracks have popularity scores
# - Diverse metadata: Artist, album, genre information included