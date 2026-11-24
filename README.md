# Customer Segmentation using K-Means Clustering

Groups 200 mall customers into five segments by annual income and spending score, then names each segment and suggests an action for it.
Built for **Task 3** of my data science internship at Arch Technologies.

![Customer segments](images/customer_segments.png)

## Problem
No column says which group a customer belongs to, so this is an **unsupervised learning** problem. The goal is to find groups of customers with similar purchasing behaviour that a mall could treat differently.

## Approach
1. Load and check the data (200 rows, no missing values, no duplicates)
2. Clean columns and explore distributions and relationships
3. Scale `Income` and `Spending` with `StandardScaler`, because K-Means works with distances
4. Choose the number of clusters with the **elbow method** and the **silhouette score**
5. Fit K-Means with K = 5 and profile each cluster
6. Name each segment and recommend an action

No train/test split is used, because there is no target to predict.

## Results
The elbow method bends at K = 5 and the silhouette score is highest there (**0.5547**), so five clusters were used.

| Segment | Customers | Profile | Suggested action |
|---|---|---|---|
| Average customers | 81 | Middle income, middle spending | Regular discounts and seasonal campaigns |
| Premium customers | 39 | High income, high spending | VIP offers and early access |
| Careful big earners | 35 | High income, low spending | Targeted promotions on premium items |
| Budget savers | 23 | Low income, low spending | Low-cost clearance offers |
| Young enthusiasts | 22 | Low income, high spending, average age 25 | Trendy products, student deals, loyalty rewards |

Income and spending are almost uncorrelated (0.01), which is why a single trend line cannot describe these customers but clustering can.

| Elbow method | Silhouette score |
|---|---|
| ![Elbow method](images/elbow_method.png) | ![Silhouette score](images/silhouette_score.png) |

## Project structure
```
customer-segmentation-kmeans/
├── data/                  # put Mall_Customers.csv here (see data/README.md)
├── images/                # charts used in this README
├── notebooks/
│   └── customer_segmentation.ipynb   # full analysis with explanations and outputs
├── src/
│   └── customer_segmentation.py      # the same analysis as a plain script
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## How to run
```bash
git clone https://github.com/<your-username>/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
pip install -r requirements.txt
# download Mall_Customers.csv into data/ (see data/README.md)
jupyter notebook notebooks/customer_segmentation.ipynb
# or run it as a script from the repository root:
python src/customer_segmentation.py
```

## Limitations
- Small dataset (200 customers)
- Only two features are used to form the segments; age and gender only describe them
- The spending score is supplied by the mall, not measured from real purchases
- K-Means needs the number of clusters chosen in advance

## Possible improvements
Add age and gender as features, compare with hierarchical clustering or DBSCAN, and use real purchase data.

## Tools
Python, pandas, matplotlib, seaborn, scikit-learn

## Author
Areej Simab
