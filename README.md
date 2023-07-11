# GRIDS: Personalized Guideline Recommendations while Driving Through a New City

Sugandh Pargal, Debasree Das, Bikash Sahoo, Bivas Mitra, Sandip Chakraborty: "GRIDS: Personalized Guideline Recommendations while Driving Through a New City", ACM Transaction on Recommender Systems 2023

GRIDS captures data from vehicle-mounted smart devices, like a dashcam with embedded sensors (IMU and GPS), which are common in today’s vehicles, and processes the data over the driver’s smartphone. The driving environments in different cities are crowdsourced using similar modalities. 

GRIDS works by the following these steps:
1. Feature Extraction: It extracts driving-related features from the raw sensor data. These features are categoriesd into four broader categories - Speed, driving 
   Maneuvers, Traffic Obstacles, Preceeding Vehicle's Action.
   Based on the speed and acceleration values, we extract statistical features and quartiles as the features. The codebase is provided in file feature extraction.py.
   Next, we utilize IMU and GPS sensors to extract 5 different driving maneuvers – (i) weaving, (ii) swerving, (iii) side-slipping, (iv) abrupt stops, and (v) sharp turns.     To detect the above features, we apply standard accelerometry analysis as adopted in literature.
   To understand the influence of traffic parameters on driving, we use the video sensor data to capture the front view from the ego vehicle. We detect the following 
   traffic objects – Pedestrians, Cars, Buses, Trucks, Bicycles, and Motorcycles. We use YOLO to detect these objects.
   We also extract four features from the preceding vehicle – Relative Distance from the preceding vehicle, Relative Speed for the preceding vehicle, the Braking Action of 
   the preceding vehicle, and the Road Congestion.

2. We rank each of the feature categories based on the driver's prior historical data as well as the current driving data in new city. This is done based on the statistical computation of L2 norm between the quartiles of the computed feature of indivisual category. The code is available in rank feature.py.
3. We train the individual feature model over Random Forest. Based on the Explainable AI, i.e the feature importance and the SHAP (Shapely Additive values), we chose the features which are required for the recommendation. (codebase Random forest.py and Explainable AI.py)
4. Finally, we utilise the rules available over web (web rules) and the nlp technique of tf-idf to generate rules for recommendation to people (codebase in nlp rule generation).

## Installation

For installation, use the following command.

```
git clone https://github.com/SUGANDHPARGAL/GRIDS.git
pip install -r requirements.txt
```

## How to Run?
```
We have provided a sample subset of our dataset in the sample data directory which consists of extracted features from different driving profiles.

Run randomforest.py to develop individual models for each profile of the driver based on his past city and present city and ExplainableAI.pynb to check the SHAP values and feature importance scores of the derived features.

Rank the category of features in order of relevance to the driver, run rank_features.py.

Finally, using the keywords of each feature from the feature category and the contextual verb from the web rules, generate rules from the existing rules using nlp. For this, run file nlp_rule_generation.py.

To make a better representation for human interpretation, GRIDS generate funnel chart from recommending_guidelines.py.


```
## Bibliography
```
 @inproceedings{spargal2023GRIDS,
  title={GRIDS: Personalized Guideline Recommendations while Driving Through a New City},
  author={Pargal, Sugandh and Das, Debasree and Sahoo, Bikash and Mitra, Bivas and Chakraborty, Sandip},
  booktitle={ACM TORS},
  year={2023}
}



