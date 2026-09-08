# GDELT GKG Feature Documentation

## Basic Count & Volume Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `article_count` | Total number of articles in 15-min interval | Count of GKGRECORDID | Measures overall news volume |
| `article_count_change` | Change from previous interval | Current count - previous count | Indicates news activity acceleration |
| `article_volume_spike` | Flag for unusual activity | 1 if count > 2 standard deviations above mean | Identifies significant news bursts |
| `prev_article_count` | Previous interval's count | Shifted article_count | Used for change calculation |

## Theme Presence Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `theme_Energy_sum` | Energy theme mentions | Sum of all Energy theme flags | Measures energy news coverage |
| `theme_Energy_max` | Energy theme presence | Max value (0 or 1) | Binary indicator of any energy news |
| `theme_Energy_mean` | Energy theme avg presence | Mean of theme flags (0-1) | Proportion of energy-themed articles |
| `theme_Environment_sum` | Environment mentions | Sum of theme flags | Weather events, climate coverage |
| `theme_Infrastructure_sum` | Infrastructure mentions | Sum of theme flags | Power grid, road, facility news |
| `theme_Social_sum` | Social events mentions | Sum of theme flags | Gatherings, festivals, protests |
| `theme_Health_sum` | Health topic mentions | Sum of theme flags | Health emergencies, epidemics |
| `theme_Political_sum` | Political mentions | Sum of theme flags | Political events, decisions |
| `theme_Economic_sum` | Economic mentions | Sum of theme flags | Financial, market, economic news |

## Tone & Sentiment Metrics

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `tone_tone_mean` | Average article tone | Mean of tone values | Overall sentiment (-100 to +100) |
| `tone_tone_min` | Most negative tone | Min of tone values | Worst news sentiment |
| `tone_tone_max` | Most positive tone | Max of tone values | Best news sentiment |
| `tone_volatility` | Tone range | tone_max - tone_min | Sentiment polarization |
| `tone_negative_max` | Max negative score | Max of negative values | Strength of negative sentiment |
| `tone_positive_max` | Max positive score | Max of positive values | Strength of positive sentiment |
| `tone_polarity_mean` | Avg opinion polarity | Mean of polarity values | How one-sided sentiment is |
| `tone_activity_mean` | Avg activity score | Mean of activity values | Active vs. passive language |

## Entity & Amount Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `entity_count_sum` | Total entities mentioned | Sum of entity counts | Scale of news coverage |
| `entity_variety_max` | Max entity variety | Max of entity_variety | Diversity of entities in news |
| `max_amount_max` | Largest amount mentioned | Max of max_amount | Scale of numeric values |

## Composite Indicators (Feature Engineering)

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `energy_crisis_indicator` | Energy crisis severity | theme_Energy_sum * tone_negative_max | Power outage/shortage intensity |
| `weather_alert_indicator` | Weather emergency level | theme_Environment_sum * abs(tone_tone_min) | Severity of weather events |
| `social_event_indicator` | Social gathering scale | theme_Social_sum * article_count / 100 | Size of social events/crowds |
| `infrastructure_stress` | Infrastructure problems | theme_Infrastructure_sum * tone_negative_max | Grid/road/facility issues |
| `political_crisis_indicator` | Political crisis level | theme_Political_sum * tone_negative_max | Political instability |
| `economic_impact_indicator` | Economic disruption | theme_Economic_sum * tone_volatility | Market/economic instability |

## Theme Interaction Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `Energy_Economic_interaction` | Energy-Economy relationship | theme_Energy_sum * theme_Economic_sum | Energy market impacts |
| `Energy_Political_interaction` | Energy-Political relationship | theme_Energy_sum * theme_Political_sum | Energy policy coverage |
| `Energy_Environment_interaction` | Energy-Environment relationship | theme_Energy_sum * theme_Environment_sum | Clean energy, climate stories |
| `Energy_Infrastructure_interaction` | Energy-Infrastructure relationship | theme_Energy_sum * theme_Infrastructure_sum | Grid issues, energy facilities |
| `Energy_negative_impact` | Negative energy news | theme_Energy_sum * tone_negative_max | Energy problems, crises |
| `Energy_volatility_impact` | Energy news volatility | theme_Energy_sum * tone_volatility | Energy market/supply uncertainty |
| `Energy_vs_Economic_ratio` | Energy relative to economy | theme_Energy_sum / (theme_Economic_sum + 0.1) | Energy focus vs economic news |
| `Energy_vs_Political_ratio` | Energy relative to politics | theme_Energy_sum / (theme_Political_sum + 0.1) | Energy focus vs political news |

## Time Context Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `hour` | Hour of day (0-23) | time_bucket.hour | Time of day effects |
| `day_of_week` | Day of week (0-6) | time_bucket.dayofweek | Weekly patterns |
| `is_weekend` | Weekend indicator | 1 if day_of_week >= 5 | Weekend vs. weekday |
| `is_business_hours` | Business hours | 1 if hour 9-17 & weekday | Work hour patterns |
| `month` | Month (1-12) | time_bucket.month | Monthly seasonality |
| `day` | Day of month (1-31) | time_bucket.day | Monthly patterns |
| `hour_sin`, `hour_cos` | Cyclical hour encoding | sin/cos transforms | Circular time patterns |
| `day_of_week_sin`, `day_of_week_cos` | Cyclical day encoding | sin/cos transforms | Circular weekly patterns |

## Temporal Effect Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `theme_Energy_lag1` | Energy theme 15min lag | theme_Energy_sum shifted 1 period | Very recent energy news |
| `theme_Energy_lag4` | Energy theme 1hr lag | theme_Energy_sum shifted 4 periods | Energy news from past hour |
| `theme_Energy_lag12` | Energy theme 3hr lag | theme_Energy_sum shifted 12 periods | Energy news from past 3 hours |
| `theme_Energy_lag24` | Energy theme 6hr lag | theme_Energy_sum shifted 24 periods | Energy news from past 6 hours |
| `theme_Energy_lag1_slow_decay` | Energy theme with slow decay | lag feature * exp(-0.01*lag) | Slowly diminishing impact |
| `theme_Energy_lag1_medium_decay` | Energy theme with medium decay | lag feature * exp(-0.05*lag) | Medium diminishing impact |
| `theme_Energy_lag1_fast_decay` | Energy theme with fast decay | lag feature * exp(-0.1*lag) | Quickly diminishing impact |
| `theme_Energy_ewma_fast` | Energy 1hr moving average | 4-period EWM | Short-term momentum (1hr) |
| `theme_Energy_ewma_medium` | Energy 6hr moving average | 24-period EWM | Medium-term momentum (6hr) |
| `theme_Energy_ewma_slow` | Energy 24hr moving average | 96-period EWM | Long-term momentum (24hr) |
| `theme_Energy_memory_effect` | Cumulative negative energy news | Decaying sum of negative energy news | Lingering effects of crises |
| `theme_Energy_memory_interaction` | Current vs accumulated effect | theme_Energy_sum * memory_effect | Reinforcing patterns |

## Time-Weighted Features

| Feature | Description | Calculation Method | Significance |
|---------|-------------|-------------------|--------------|
| `theme_Energy_business_impact` | Business hours energy impact | theme_Energy_sum * business_hours_weight | Impact during working hours |
| `theme_Energy_weekend_adjusted` | Weekend energy impact | theme_Energy_sum * weekend_weight | Impact during residential peak |
| `theme_Energy_evening_impact` | Evening peak energy impact | theme_Energy_sum * evening_peak_weight | Impact during evening demand peak |
| `theme_Energy_temporal_weighted` | Combined temporal impact | theme_Energy_sum * combined_weight | Overall temporally-weighted impact |

*Features marked with asterisk (*) may be removed in feature selection due to high correlation with other features.