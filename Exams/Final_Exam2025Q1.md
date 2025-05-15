## Final Exam for the Artificial Intelligence & Python Programming (2024 Spring)
## Topics: Hackathon
Lei Ge 5/14/2025
#### Why `Hackathon`?

Hackathon definition
A hackathon is an event set up by a company or an organization that wants to get a high-quality solution through collaboration between experts. Hackathon format is often competitive.

For example, an organization wants to design a brand new operating system. It hosts a hackathon that brings together 10 startups or teams of developers.  Each team provides a solution. The jury chooses the best product and hands out a prize.  After the event, the organization may choose to sign a contract with the winning team. 

It's not just the winner, who gets a shot at landing a contract. Many tech companies monitor hackathon activities and buy products or choose new team members on the spot.

reference: [https://tips.hackathon.com/article/what-is-a-hackathon]




#### I.  Quant Modeling Project 

1. Train and test data split 
  - you should also address the midterm exam comments from me and your validation team
  - Any models such as Linear Regression, Lasso, Random Forest, Xgboost, Keras ANN to train you model by using the training sample
  - use sklearn.inspection.permutation_importance or other feature importance method to get the feature importance table for the final model
  - Model performance table and feature importance table are necessary (pdp not required but maybe useful)
- Model Performance Table Example (model performance metrics: rmse):
| Metrics| Hackthon Score| In sample | Cross-validation | Testing | Total N of testing set after dropping outliers|
| --- | --- | --- | --- |  --- |--- |
| Best Model | 78.45 |12345 | 12345 |12345 | 2000|



#### II. Individual Presentation for Innovations (`June 5th and June 12th)

- Presentation is important, it is a simulation of the job interview for the quant researcher or economist position or your future academic works
- You should **highline** the `innovations` during the hackathon presentation 
- Your score `only depends` on the contents and innovations mentioned by your presentations, but your presentation should be backed by your codes and results (!!!Your results will be checked during the presentation!!!)
- Only talk key points
- max 6 slides 
- `10 min` presentation and `2-3 questions` to check your works, innovations and other key points 
- penalties for the long presentation: up to 5% off 



#### IIIa. Hackathon all students first result submission (`submission due June 4th` one day before final)

  - Codes: `FinalExam_First_studentID.ipynb` (datahub: https://datahub.ruc.edu.cn/org/RUC/competition/area/67ddbdc89f172c5665cb4de0/submit)
  - Scoring: `prediction.csv` (datahub)
  - Slides: `presentation.pdf` (GitHub)
  - This will the final submission for the graduating students 



#### IIIb. Hackathon final result submission (`submission due June 15th`)

  - Codes: `FinalExam_Second_studentID.ipynb` (datahub: https://datahub.ruc.edu.cn/org/RUC/competition/area/67ddbdc89f172c5665cb4de0/submit)
  - Scoring: `prediction.csv` (datahub)
  - Slides: `presentation.pdf` (Github)



### IV. Scoring


1. Grading Progress
- 1st. line review on the presentation and questions (Lei Ge) 
- 2nd. line review of codes and questions (model validators: Xiangyuan Mo, Chenxi Wang, Tianyou Cui, Yichen Xu) aka model validation
- 3rd. line review of codes (model auditor: Lei Ge) aka model auditing

2. key points of the grading:

- The `ML techniques and strategies` during the presentations
- `Innovations` (!!! Important !!!)
- `formality of coding`
- `Bonus to try new`:
  - pca, autoencoder, 
  - optuna, hyperopt
  - Bert, Roberta, deepseek or other NPL 
  - GNN, CNN, RNN
  - Transfer learning 
  - pretraining
  - external data 
  - all interesting innovations

   
