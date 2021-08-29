import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd
pd.set_option('display.width', 1024)

#A subset of health tips for demo purpose only
data = {"health_note":
    {
 0: "Lose extra weight. Moving toward a healthy weight helps control blood sugars. Your doctor, a dietitian, and a fitness trainer can get you started on a plan that will work for you."
,1: "Check your blood sugar level at least twice a day. Is it in the range advised by your doctor? Also, write it down so you can track your progress and note how food and activity affect your levels."
,2: "Get A1c blood tests to find out your average blood sugar for the past 2 to 3 months. Most people with type 2 diabetes should aim for an A1c of 7% or lower. Ask your doctor how often you need to get an A1c test."
,3: "Track your carbohydrates. Know how many carbs you’re eating and how often you have them. Managing your carbs can help keep your blood sugar under control. Choose high-fiber carbs, such as green vegetables, fruit, beans, and whole grains."
,4: "Control your blood pressure, cholesterol, and triglyceride levels.Diabetes makes heart disease more likely, so keep a close eye on your blood pressure and cholesterol. Talk with your doctor about keeping your cholesterol, triglycerides, and blood pressure in check. Take medications as prescribed."
,5: "Keep moving. Regular exercise can help you reach or maintain a healthy weight. Exercise also cuts stress and helps control blood pressure, cholesterol, and triglyceride levels. Get at least 30 minutes a day of aerobic exercise 5 days a week. Try walking, dancing, low-impact aerobics, swimming, tennis, or a stationary bike. Start out more slowly if you aren't active now. You can break up the 30 minutes -- say, by taking a 10-minute walk after every meal. Include strength training and stretching on some days, too."
,6: "Catch some ZZZs. When you’re sleep-deprived, you tend to eat more, and you can put on weight, which leads to health problems. People with diabetes who get enough sleep often have healthier eating habits and improved blood sugar levels."
,7: "Manage stress. Stress and diabetes don't mix. Excess stress can elevate blood sugar levels. But you can find relief by sitting quietly for 15 minutes, meditating, or practicing yoga."
,8: "See your doctor. Get a complete checkup at least once a year, though you may talk to your doctor more often. At your annual physical, make sure you get a dilated eye exam, blood pressure check, foot exam, and screenings for other complications such as kidney damage, nerve damage, and heart disease."
    }
}


print('Reading in data ...')
df = pd.DataFrame(data)

print('Building TF IDF..')

#Instantiate TF-IDF Vectorizer Object. Remove all english stopwords
tfidf = TfidfVectorizer(stop_words='english')

df['health_note'] = df['health_note'].fillna('')
df['item_id'] = df.index

#applying fit_transform
tfidf_matrix = tfidf.fit_transform(df['health_note'])

#Construct a mapping of indices and item_ids, and drop duplicate item_ids
indices = pd.Series(df.index, index=df['item_id']).drop_duplicates()

print('Computing cosine similarity ..')
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

#Define function taking in item_id as input and spit out top k recommendations 
def content_recommender(item_id, k, cosine_sim=cosine_sim, df=df, indices=indices):
    print("Given item id %d - %s\n" %(item_id, df['health_note'].iloc[item_id]))
    
    # Obtain the index of the item that matches the item_id
    idx = indices[item_id]

    # Get the pairwsie similarity scores of all items with that item
    # And convert into a list of tuples 
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the items based on cosine similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the k most similar items. 
    sim_scores = sim_scores[1:k+1]

    # Get the item indices
    my_indices = [i[0] for i in sim_scores]

    # Return the top k most similar items
    return df['health_note'].iloc[my_indices] 

if __name__ == "__main__":
    print('Making recommendations given an exisiting item...')
    print(content_recommender(3, 10))

