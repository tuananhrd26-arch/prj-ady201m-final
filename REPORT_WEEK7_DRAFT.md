# III. Week 7 Data Analysis Results

This section presents the Week 7 analytical results for the Spotify Recommendation Analytics project. The analysis combines SQL-based querying, regression modeling, and Python-based visualization to explain temporal patterns, feature relationships, and popularity prediction behavior in the cleaned Spotify dataset.

## 3.1 SQL-Based Data Analysis

### Purpose of Analysis

The purpose of the SQL-based analysis is to transform the cleaned Spotify CSV files into structured analytical outputs using database-style queries. SQL is used to aggregate tracks by decade, summarize audio feature trends, rank top tracks within each decade, identify tracks with above-average popularity, and compare artists and genres based on popularity and audio characteristics. This approach supports reproducible analysis because each table is generated from explicit SQL queries saved in the project output folder.

### Output Files to Insert in the Report

Use the following files from `week7_outputs\sql`:

```text
week7_outputs\sql\spotify_week7_queries.sql
week7_outputs\sql\01_tracks_by_decade.csv
week7_outputs\sql\02_audio_features_by_year.csv
week7_outputs\sql\03_top_tracks_by_decade_window.csv
week7_outputs\sql\04_above_average_popularity_subquery.csv
week7_outputs\sql\05_top_artists_by_popularity.csv
week7_outputs\sql\06_top_genre_audio_profiles.csv
```

Recommended supporting table:

```text
week7_outputs\tables\tracks_by_decade.csv
```

### Academic Insight Explanation

The SQL results show a clear temporal pattern in Spotify track popularity. Average popularity increases substantially across decades, from approximately 1.30 in the 1920s to 64.30 in the 2020s. This trend suggests that more recent songs tend to receive higher popularity scores in the dataset. The pattern may reflect contemporary listening behavior, platform recency effects, and the fact that recent tracks are more likely to be actively streamed by current Spotify users.

The decade-level aggregation also indicates important changes in audio characteristics. Average energy increases from 0.235 in the 1920s to 0.631 in the 2020s, while average danceability rises to 0.693 in the 2020s. These results suggest that modern popular music in the dataset is generally more energetic and dance-oriented than earlier recordings. At the same time, the number of tracks in the 2020s is lower than previous full decades because the dataset only covers the early part of that decade.

The SQL ranking outputs provide additional interpretability. The window-function query identifies the top tracks within each decade, allowing popularity to be compared within historical periods rather than only across the whole dataset. The above-average popularity query shows that many of the largest positive popularity gaps belong to 2020 tracks such as "Dakiti", "Mood", "Dynamite", and "Blinding Lights". This supports the broader conclusion that recent releases dominate the highest popularity values.

The artist and genre aggregation queries also reveal that high-popularity groups tend to be associated with contemporary pop, Latin, dance, and hip-hop-influenced music. For example, the top genre profile table includes genres such as basshall, trap venezolano, south african house, and turkish edm, many of which have high danceability and relatively low acousticness. This reinforces the observation that current popularity is connected not only to release period but also to energetic and rhythm-driven audio profiles.

### Suggested Placement for Tables

Insert `01_tracks_by_decade.csv` immediately after the paragraph discussing decade-level popularity. This table should be labeled as a summary of track count, average popularity, energy, and danceability by decade.

Insert `03_top_tracks_by_decade_window.csv` after explaining SQL window functions. This table can demonstrate how SQL ranks tracks within each decade.

Insert `05_top_artists_by_popularity.csv` and `06_top_genre_audio_profiles.csv` near the end of this subsection to support the artist and genre comparison discussion.

Mention `spotify_week7_queries.sql` in the methodology paragraph as evidence that the SQL analysis is reproducible.

## 3.2 Regression Analysis

### Purpose of Analysis

The purpose of the regression analysis is to evaluate how well Spotify audio features and metadata can explain track popularity. The target variable is `popularity`, while the predictors include acousticness, danceability, energy, instrumentalness, liveness, loudness, speechiness, tempo, valence, duration, year, explicit, key, and mode. Two models are compared: Linear Regression and Ridge Regression. Ridge Regression is included because it provides coefficient regularization and is useful when predictors are correlated.

### Output Files to Insert in the Report

Use the following files from `week7_outputs\tables`:

```text
week7_outputs\tables\regression_metrics.csv
week7_outputs\tables\regression_coefficients.csv
week7_outputs\tables\regression_actual_vs_predicted.csv
```

Use the following figures from `week7_outputs\figures`:

```text
week7_outputs\figures\regression_actual_vs_predicted.png
week7_outputs\figures\regression_residuals.png
week7_outputs\figures\regression_coefficients.png
```

Recommended supporting visualization:

```text
week7_outputs\figures\correlation_heatmap.png
```

### Academic Insight Explanation

The regression results show that both Linear Regression and Ridge Regression produce almost identical performance. Linear Regression achieves an MAE of 7.982, RMSE of 10.7308, and R2 of 0.7594. Ridge Regression achieves an MAE of 7.9821, RMSE of 10.7309, and R2 of 0.7594. These results indicate that the selected features explain approximately 75.94 percent of the variance in track popularity. The similarity between the two models suggests that regularization does not substantially change predictive performance for this feature set.

The coefficient analysis shows that `year` is by far the strongest positive predictor of popularity, with a Ridge standardized coefficient of approximately 17.286. This means that newer tracks are strongly associated with higher predicted popularity after controlling for the other included features. This finding is consistent with the SQL analysis, where average popularity rises across decades and reaches its highest value in the 2020s.

Among audio features, acousticness, instrumentalness, and speechiness have the strongest negative coefficients. Ridge Regression estimates acousticness at approximately -1.574, instrumentalness at -1.327, and speechiness at -1.257. This suggests that, within this model, tracks with more acoustic, instrumental, or speech-heavy profiles tend to receive lower predicted popularity scores. In contrast, danceability and explicit content have positive coefficients, although their effects are much smaller than the effect of year.

The actual-versus-predicted plot shows that the model captures the general upward relationship between actual and predicted popularity. However, the plot also suggests that predictions are compressed for highly popular tracks. In other words, the model is effective at identifying broad popularity patterns but is less precise for extreme popularity values. The residual plot further indicates that errors are not perfectly random, partly because popularity is bounded between 0 and 100 and many tracks have very low or zero popularity. Therefore, the regression model is useful for explanation and baseline prediction, but more advanced models may be needed for highly accurate prediction of top-charting songs.

### Suggested Placement for Tables and Figures

Insert `regression_metrics.csv` first in this subsection, immediately after introducing the two regression models. This table should be used to compare MAE, RMSE, and R2.

Insert `regression_coefficients.csv` or `regression_coefficients.png` after discussing feature importance. The figure is more suitable for the main report, while the CSV can be referenced as a detailed appendix table.

Insert `regression_actual_vs_predicted.png` after explaining the model's predictive performance.

Insert `regression_residuals.png` after discussing model limitations and residual behavior.

Insert `correlation_heatmap.png` before or after the coefficient discussion to connect correlation analysis with regression findings.

## 3.3 Data Analysis Tools and Visualization

### Purpose of Analysis

The purpose of this subsection is to explain how Python-based data analysis tools were used to support exploratory data analysis, visualization, modeling, and recommendation demonstration. The analysis pipeline uses pandas and NumPy for data processing, SQLite for SQL execution, matplotlib for visualization, scikit-learn for regression and nearest-neighbor modeling, and joblib for saving model artifacts. These tools make the analysis reproducible and allow both statistical outputs and visual evidence to be generated from the same cleaned dataset.

### Output Files to Insert in the Report

Use the following summary tables from `week7_outputs\tables`:

```text
week7_outputs\tables\dataset_overview.csv
week7_outputs\tables\descriptive_statistics.csv
week7_outputs\tables\missing_values_after_cleaning.csv
week7_outputs\tables\audio_feature_trends_by_year.csv
week7_outputs\tables\correlation_matrix.csv
week7_outputs\tables\top_genres_audio_profile.csv
week7_outputs\tables\recommendation_demo_results.csv
```

Use the following figures from `week7_outputs\figures`:

```text
week7_outputs\figures\tracks_by_decade.png
week7_outputs\figures\audio_feature_trends_by_year.png
week7_outputs\figures\correlation_heatmap.png
```

Optional model artifacts for technical appendix:

```text
week7_outputs\model_artifacts\ridge_popularity_model.joblib
week7_outputs\model_artifacts\nearest_neighbors_recommender.joblib
week7_outputs\model_artifacts\recommender_scaler.joblib
week7_outputs\model_artifacts\recommender_features.json
```

### Academic Insight Explanation

The visualization outputs provide a clearer understanding of the dataset before and after modeling. The dataset overview table confirms that the main tracks table contains 170,653 records and 20 columns, while additional artist, genre, year, and artist-genre tables provide complementary analytical dimensions. This confirms that the analysis is based on a sufficiently large dataset for exploratory and regression-based investigation.

The track count by decade visualization shows that the dataset contains relatively fewer tracks in the earliest decades and in the 2020s, while the period from the 1950s to the 2010s contains close to 20,000 tracks per decade. This is important because visual interpretation must consider dataset coverage. The lower count in the 2020s does not necessarily indicate lower music production; it reflects that the decade is incomplete in the dataset.

The audio feature trend visualization shows a long-term transformation in musical characteristics. Acousticness declines sharply over time, while energy and danceability increase. This suggests a shift from more acoustic historical recordings toward more electronically produced, energetic, and rhythm-centered modern tracks. The pattern is also consistent with the regression and SQL results, where recent and energetic music profiles are more strongly associated with popularity.

The correlation heatmap supports the feature interpretation used in the regression section. Popularity is positively correlated with energy and loudness, while it is negatively correlated with acousticness and instrumentalness. Energy and loudness are also strongly correlated with each other, which is expected because louder tracks often have higher perceived intensity. This correlation structure helps explain why Ridge Regression is a reasonable comparison model, even though its final performance is similar to ordinary Linear Regression.

The recommendation demo table illustrates how nearest-neighbor similarity can be used to generate content-based recommendations from audio features. However, the output should be interpreted as a technical demonstration rather than a final user-facing recommendation system. Some recommendations may be musically similar according to numerical audio features but culturally or contextually distant from the input song. This limitation shows that a more complete recommendation system should combine audio similarity with artist, genre, language, release period, and user behavior data.

### Suggested Placement for Tables and Figures

Insert `dataset_overview.csv` at the beginning of this subsection to document the scale of the data used.

Insert `tracks_by_decade.png` after describing dataset coverage across decades.

Insert `audio_feature_trends_by_year.png` after discussing long-term changes in acousticness, energy, danceability, and valence.

Insert `correlation_heatmap.png` before the transition into regression analysis, or use it as a bridge between visualization and modeling.

Insert `top_genres_audio_profile.csv` near the genre discussion to show how high-popularity genres differ by danceability, energy, acousticness, valence, and tempo.

Insert `recommendation_demo_results.csv` near the end of the subsection as evidence of the content-based recommendation demonstration.

## Recommended Figures and Tables for the Final Report

For the main body of the report, the most useful outputs are:

```text
week7_outputs\sql\01_tracks_by_decade.csv
week7_outputs\tables\regression_metrics.csv
week7_outputs\figures\regression_actual_vs_predicted.png
week7_outputs\figures\regression_coefficients.png
week7_outputs\figures\audio_feature_trends_by_year.png
week7_outputs\figures\correlation_heatmap.png
week7_outputs\tables\top_genres_audio_profile.csv
```

For the appendix, include:

```text
week7_outputs\sql\spotify_week7_queries.sql
week7_outputs\tables\regression_coefficients.csv
week7_outputs\tables\recommendation_demo_results.csv
week7_outputs\sql\03_top_tracks_by_decade_window.csv
```
