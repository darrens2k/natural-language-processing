import pandas as pd
import spacy

# paths to files
data_path = 'A3_reviews.csv'
ner_path = 'model-best'

# read in data
data = pd.read_csv(data_path, sep='\t')
print(data.info())

# create a list of only reviews
reviews = data['Review'].tolist()
print(reviews[0])

# use spacy to load in the NER model
ner = spacy.load(ner_path)

# run the ner model on the reviews
# must iterate through each review
# begin with only the first 5 reviews as a test
for i in range(2):

    # empty list to store the dishes in the review
    dishes = []

    # run model on the review
    review_out = ner(reviews[i])

    # grab all of the entities the model found in the review
    entities = review_out.ents

    # check if any of the entities were assigned the label dish
    for entity in entities:

        # extract the labels the model assigned to the entities
        if entity.label_ == 'DISH':

            # if a dish was found, add the tags around the dish in that review
            dishes.append(entity.text)

    # remove any duplicate dishes
    dishes = list(set(dishes))

    # sort dishes in descending order, reason inside loop comments
    dishes = sorted(dishes, key=len, reverse=True)
    print(dishes)
    # iterate through all dishes
    for dish in dishes:

        # need to make sure we are not within another 
        # for example, the code could think 'taco' within 'lobster tacos' is another dish
        # to stop this check if a start tag with no end tag exists
        # longer dishes must come first for this to work

        current_review = reviews[i]
        rev = current_review
        start_index = 0
        print(dish)
        while True:
            
            locate = current_review.find(dish, start_index)
            # print(locate)
            if locate == -1 or start_index >= len(reviews[i]) - 1:
                print("broke")
                break

            

            # split the review into 2 list elements at the dish
            first, second = rev.split(dish, 1)
            
            # check for the first tag to appear before the dish
            # if it is an end tag, proceed, if it is start tag: we're in the middle of another dish
            last_start_tag_loc = first.rfind("[B-ASP]")
            last_end_tag_loc = first.rfind("[E-ASP]")

            # check if a start tag is located before an end tag
            if last_start_tag_loc > last_end_tag_loc:

                # do not continue the loop, do not update the review here
                continue
            

            # add tags into the review
            reviews[i] = reviews[i].replace(dish, f"[B-ASP]{dish}[E-ASP]")

            start_index = locate + len(dish)
            print(start_index)
            rev = current_review[start_index:]


print(reviews[0])

