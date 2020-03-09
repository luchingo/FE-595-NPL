import ast
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as senti

# Part1: Creating two lists to store the normalized values

my_names = []
my_purpose = []

# File 1:

file = open("WS1.txt", "r")
contents = file.read()

dict_1 = ast.literal_eval(contents)
file.close()

for key in dict_1.keys():
    my_purpose.append(dict_1[key])
    my_names.append(key)

# File 2:

tab2 = pd.read_csv("WS2.csv")

for i in range(len(tab2["Name"])):
    my_names.append(tab2["Name"][i])
    my_purpose.append(tab2["Purpose"][i])

# File 3:

tab3 = pd.read_csv("WS3.txt", sep=":", header=None)

tab3[1] = tab3[1].str.rstrip(", Purpose")

for i in range(len(tab3[1])):
    my_names.append(tab3[1][i])
    my_purpose.append(tab3[2][i])

# File 4:
tab4 = pd.read_csv("WS4.txt", sep=":", header=None)

my_list_1 = list(tab4.iloc[::2, 1])
my_list_2 = list(tab4.iloc[1::2, 1])

for i in range(len(my_list_1)):
    my_names.append(my_list_1[i])
    my_purpose.append(my_list_2[i])

# File 5:
tab5 = pd.read_csv("WS5.txt", sep=":", header=None)

my_list_3 = list(tab5[1][:50])
my_list_4 = list(tab5[1][50:])

for i in range(len(my_list_3)):
    my_names.append(my_list_3[i])
    my_purpose.append(my_list_4[i])

# File 6:

tab6 = pd.read_csv("WS6.csv")

for i in range(len(tab6["name"])):
    my_names.append(tab6["name"][i])
    my_purpose.append(tab6["purpose"][i])

# File 7:
tab7 = pd.read_csv("WS7.txt", sep=":", header=None)

my_list_7 = list(tab7.iloc[::2, 1])
my_list_8 = list(tab7.iloc[1::2, 1])

for i in range(len(my_list_7)):
    my_names.append(my_list_7[i])
    my_purpose.append(my_list_8[i])

# File 8:
tab8 = pd.read_csv("WS8.csv")

for i in range(len(tab8["name"])):
    my_names.append(tab8["name"][i])
    my_purpose.append(tab8["purpose"][i])

# File 9:
tab9 = pd.read_csv("WS9.txt", sep="\t")

for i in range(len(tab9["Name"])):
    my_names.append(tab9["Name"][i])
    my_purpose.append(tab9["Purpose"][i])

# File 10:

tab10 = pd.read_csv("WS10.csv")

for i in range(len(tab10["Name"])):
    my_names.append(tab10["Name"][i])
    my_purpose.append(tab10["Purpose"][i])


# Part 2: Sentiment analysis

# First I remove any possible duplicate values. I do so y transforming the list values into keys of a dictionary
# and then transforming them back to a list

score_data = pd.DataFrame()
score_data["Name"] = my_names
score_data["Purpose"] = my_purpose

scores = []
for i in range(len(my_purpose)):
    my_score = senti().polarity_scores(my_purpose[i])
    scores.append(my_score["compound"])

score_data["Score"] = scores

score_data = score_data.sort_values(by="Score", ascending=False)

score_data.nlargest(10, "Score")
score_data.nsmallest(10, "Score")
