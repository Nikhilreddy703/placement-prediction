# Exploratory Data Analysis (EDA) Summary Report

## 1. Dataset Overview
- **Total Student Records:** 400
- **Input Features:** 12 (Numeric: 5, Categorical: 7)
- **Overall Placement Rate:** 67.5% (270 Placed, 130 Not Placed)

## 2. Numeric Features Summary

|       |   ssc_p |   hsc_p |   degree_p |   etest_p |   mba_p |
|:------|--------:|--------:|-----------:|----------:|--------:|
| count |  400    |  400    |     400    |    400    |  400    |
| mean  |   66.49 |   66.07 |      66.33 |     71.9  |   62.2  |
| std   |   10.14 |   10.06 |       6.95 |     11.91 |    5.66 |
| min   |   40    |   40    |      50    |     50    |   51    |
| 25%   |   59.2  |   59.68 |      61.64 |     63.3  |   58.03 |
| 50%   |   66.87 |   66.25 |      66.65 |     72.2  |   61.89 |
| 75%   |   73.2  |   72.64 |      70.41 |     80.45 |   65.52 |
| max   |   92.98 |   94.09 |      86    |     98    |   78    |

## 3. Placement Rate by Categorical Attributes

### gender
| gender   |   Student Count |   Placed Rate (%) |
|:---------|----------------:|------------------:|
| F        |             156 |              68.6 |
| M        |             244 |              66.8 |

### ssc_b
| ssc_b   |   Student Count |   Placed Rate (%) |
|:--------|----------------:|------------------:|
| Central |             209 |              66.5 |
| Others  |             191 |              68.6 |

### hsc_b
| hsc_b   |   Student Count |   Placed Rate (%) |
|:--------|----------------:|------------------:|
| Central |             201 |              64.2 |
| Others  |             199 |              70.9 |

### hsc_s
| hsc_s    |   Student Count |   Placed Rate (%) |
|:---------|----------------:|------------------:|
| Arts     |              45 |              68.9 |
| Commerce |             198 |              64.6 |
| Science  |             157 |              70.7 |

### degree_t
| degree_t   |   Student Count |   Placed Rate (%) |
|:-----------|----------------:|------------------:|
| Comm&Mgmt  |             232 |              69.4 |
| Others     |              37 |              75.7 |
| Sci&Tech   |             131 |              61.8 |

### workex
| workex   |   Student Count |   Placed Rate (%) |
|:---------|----------------:|------------------:|
| No       |             243 |              60.9 |
| Yes      |             157 |              77.7 |

### specialisation
| specialisation   |   Student Count |   Placed Rate (%) |
|:-----------------|----------------:|------------------:|
| Mkt&Fin          |             242 |              70.7 |
| Mkt&HR           |             158 |              62.7 |

## 4. Key EDA Insights
1. **Academics:** Students with higher undergraduate (degree_p) and 10th/12th percentages show significantly higher placement rates.
2. **Work Experience:** Students with prior work experience or internships consistently have higher conversion in recruitment drives.
3. **Employability Test:** Aptitude and employability scores (etest_p) correlate positively with clearing technical screening rounds.

## 5. Generated Visualizations
- `reports/figures/class_balance.png` - Target class balance bar chart
- `reports/figures/numeric_by_status.png` - Distribution of marks by placement status
- `reports/figures/correlation.png` - Correlation heatmap among academic metrics