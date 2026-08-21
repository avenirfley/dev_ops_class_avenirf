# Model Card

## Model Details

For this project, I built a binary Random Forest classifier using scikit-learn
1.6.1. I configured the model with 200 trees, a minimum leaf size of two,
balanced class weights, and a random seed of 42 so that my results are
reproducible. The model predicts whether a person's annual income is greater
than $50,000.

## Intended Use

I created this model to demonstrate what I learned about building a reproducible
machine learning pipeline and serving predictions through a FastAPI
application. I did not design or validate it for real-world decision-making. It
should not be used to make employment, lending, insurance, benefits, or other
decisions that could materially affect a person.

## Training Data

I trained the model with the provided Census Income dataset, which contains
32,561 records. The target variable includes 24,720 records labeled `<=50K` and
7,841 records labeled `>50K`. I cleaned extra whitespace from the column names
and text values before processing the data. I then used a stratified 80/20 split
with a random seed of 42, which placed 26,048 records in the training set. I
one-hot encoded the categorical features and converted the salary target into a
binary label.

## Evaluation Data

I evaluated the model with the remaining 6,513 records from the stratified
split. I did not use these records to train the model or fit the categorical
encoder. I also measured performance for each distinct value of the categorical
features so I could examine how the model performs on different groups. I saved
those results in `slice_output.txt`.

## Metrics

I used precision, recall, and the F1 score to evaluate the model, with `>50K`
treated as the positive class. Precision tells me how many of the records that
the model predicted as `>50K` were actually in that class. Recall tells me how
many of the actual `>50K` records the model identified. The F1 score gives me a
single value that balances precision and recall.

On my held-out evaluation set of 6,513 records, the model achieved a precision
of **0.6208**, a recall of **0.8259**, and an F1 score of **0.7088**. These
results mean that approximately 62.08% of the model's positive predictions were
correct. The model also identified approximately 82.59% of the records that
actually belonged to the `>50K` class.

## Ethical Considerations

I recognize that this dataset reflects historical socioeconomic patterns and
contains sensitive attributes such as age, race, and sex. Because of this, the
model could reproduce or amplify inequities that are present in the data. The
overall metrics also do not guarantee that the model performs equally well for
every demographic group. In addition, the salary threshold is dated and does
not account for inflation, location, household circumstances, or changes in the
labor market.

## Caveats and Recommendations

I developed this model as an educational project, not as a validated production
system. I recommend reviewing `slice_output.txt` because the model's performance
may vary across different groups. The predictions show associations in the
data, but they do not prove that any feature causes a person's income level. If
I were preparing this model for production, I would use more current and
representative data, perform a detailed fairness evaluation, monitor uncertainty
and data drift, complete privacy and stakeholder reviews, and retrain the model
regularly.