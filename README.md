# Results of Hackathon: **Ranking #1** 🥇 
![First Place](https://github.com/GVRQ/J2D_Data-Science_2022/blob/main/img/rank.png?raw=true)


<h3 style="text-align: left;" align="left">Connect with me:</h3>
<p style="text-align: left;" align="left"><a href="https://t.me/gavrilov_se" target="blank"><img style="float: left;" src="https://www.svgrepo.com/show/349527/telegram.svg" alt="Telegram_Alexander_Gavrilov_Data_Scientist" width="40" height="30" align="center" /></a>&nbsp;<a href="mailto:alexander@gavrilov.se" target="blank"><img src="https://www.clipartmax.com/png/full/91-913506_computer-icons-email-address-clip-art-icon-email-vector-png.png" alt="Email_Alexander_Gavrilov_Data_Scientist" width="30" height="30" align="center" /></a>&nbsp; <a href="https://www.linkedin.com/in/GVRQ/" target="blank"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/81/LinkedIn_icon.svg/72px-LinkedIn_icon.svg.png" alt="Linkedin_Alexander_Gavrilov_Data_Scientist" width="30" height="30" align="center" /></a></p>

# Air quality classification

# Background
The Paris Agreement is an international treaty on climate change that was adopted by 196 Parties at COP21 in Paris. Its goal is to limit global warming to well below 2, preferably 1.5 degrees Celsius, compared to pre-industrial levels. To reach this long-term temperature goal, countries aim to peak greenhouse gas emissions as soon as possible to achieve a climate-neutral planet by mid-century. That is why the European Union is allocating large amounts of resources to the development of new technologies that allow the improvement of the fight against pollution. One of these is a new type of sensor based on laser technology that allows air quality to be detected based on different measurements.

We have two datasets (`train.csv`,`test.csv`) with two variables:

- **Features**: The dataset contains 8 features in 8 columns, which are the parameters measured by the different sensors. These correspond to the different interactions that the laser beams have had when passing through the air particles.

- **Target**: The target corresponds to the 'label' that classifies the quality of the air.

* Target 0 = Good air quality
* Target 1 = Moderate air quality
* Target 2 = Dangerous air quality

**Datasets**:
- train.csv: This dataset contains both the predictor variables and the type of air quality classification.
- test.csv: This dataset contains the predictor variables with which the type of air quality will have to be predicted.

# Problem

In order to predict the type of air quality in the `test`-dataset, we are going to make a predictive model using ***Random Forest***.

# Results

The results are in the 'predictions.csv' file.

Model: Random Forest with optimizations.
The best result obtained with the selected model after training several models is 0.9 f1_score.

![alt text](https://github.com/GVRQ/J2D_Data-Science_2022/blob/main/img/1.png?raw=true)
![alt text](https://github.com/GVRQ/J2D_Data-Science_2022/blob/main/img/2.png?raw=true)

# Analysis

The dataset contains 8 features. 

- We've analyzed Target. The target is balanced. 
![alt text](https://github.com/GVRQ/J2D_Data-Science_2022/blob/main/img/3.png?raw=true)

- Analyzed Correlations between features
![alt text](https://github.com/GVRQ/J2D_Data-Science_2022/blob/main/img/5.png?raw=true)

- Analyzed Features Importance
![alt text](https://github.com/GVRQ/J2D_Data-Science_2022/blob/main/img/4.png?raw=true)


# Solution

After analyzing Correlations between features, we detected high correlation between Feature 5 & 6. We keep both because deleting one of them results in worse predictions. After analyzing Features Importance, we detected most important features: 3 & 6 and least important: 7 & 8. Features 7 & 8 were removed in order to reduce noise.

**Params:**
- RandomForestClassifier(random_state = 1990)
- 'bootstrap': True,
- 'criterion': 'gini',
- 'max_depth': 16,
- 'max_features': 3,
- 'max_leaf_nodes': 128,
- 'n_estimators': 256,
- 'n_jobs': 4, 
- 'cv': 5,
- 'verbose': 4,

# License
