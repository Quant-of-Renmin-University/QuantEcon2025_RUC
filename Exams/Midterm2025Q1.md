# Midterm Project: Machine Learning Modeling for Modelers   
`Report and presentation both due on April 3rd 2025 on Class`  

#### I. Why Real Modeling?
  - As a budding modeler with exceptional talent, the midterm exam serves as a dynamic simulation of your future journey as a quant economist or data scientist. This is your opportunity to craft innovative models and showcase your analytical prowess. The insights derived from your quantitative modeling will not only demonstrate your skills but also illuminate the path toward a thriving and impactful career in the field. Embrace this challenge—it’s a stepping stone to your future success.   


#### II. Workflow
1. Project : Machine Learning Modeling    

2. Framework :
  - data location: https://datahub.ruc.edu.cn/org/RUC/task/67bff956aa62251c80841535/67bff956aa62251c80841556/manage
  - data analysis, data clean and data preprocessing (Please refer to Ch1 and Ch2 codes from our textbook and my slides)  
  - delete the features with data leakage issues such as community price in the housing project
<div class="alert alert-block alert-danger">
<b>Caution:</b> No Data Leakage Here
</div>  
  - divide the sample into 80% training and 20% testing by using sklearn.model_selection.train_test_split (random_state==111)
  - You can only use linear models: **Linear Regression, Lasso, Ridge regression, Elastic Net for now**
  - Fine-tune model: try to improve your model by 1) add and drop features, 2) add non-linearity and interactions, 3) change hyperparameters (L1 and L2 regularizers) of model 
  
  - Report both your: in-sample, out of sample and [6 folds cross validation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_validate.html) model performance for your OLS, LASSO, Ridge and best model (please use Mean Absolute Error, RMAE to report the performance). Also, please report the total number of predictions after you remove the outlier of the sample.  

  
- `In your presentation you need to show the metrics table below`:  

| Metrics| In sample | out of sample | Cross-validation |Datahub Score |
| --- | --- | --- | --- | --- | 
| OLS | 0.94 |0.92 | 0.92 | 60 |
| LASSO | 0.94 |0.92 | 0.92 |61 |
| Best Model | 0.94 |0.92 | 0.92 |62 |

Note: `Metrics should be MAE, RMSE for the original housing **price level**`  

   
#### II. Submission      
   
1. Git submission (individual, before presentation April 3rd):   
    - Codes: **Midterm_codes_StudentID.ipynb**   

2. Git submission (in group folder, due April 10th): 
    - Merge your codes into one and for the next homework model validation
    - Codes: **Midterm_codes_Team1a.ipynb**     

    
3. datahub submission (Individual)
    - 
    - Scoring: **prediction.csv**
    - **!!!This score is not for midterm grading!!!**, `your presentation and modeling matters`
   
#### III. Real Presentation
7. Presentation (**!!!Important!!!**): 
- `Presentation is important`, it is a simulation of the job interview for the quant researcher, data scientist, and economist position (As young talents, please show your talents)
- You should **highline** the `innovations` during the presentation 
- Your score `only depends` on the contents and innovations mentioned by your presentations, but your presentation should be backed by your codes and results (!!!Your results should be checked during the presentation!!!)
- Only talk keys points
- Only Linear Algorithms in midterm (OLS, LASSO, Ridge and etc., but you can use fancy feature engineering)
- max 3 slides (not counts on the frontpage)
- `4 min` presentation and `2 questions` to check your works, innovations and other key points 
- penalties for the long presentation or slides: up to 5% off  




- Due date: April 3rd 2025
-------

