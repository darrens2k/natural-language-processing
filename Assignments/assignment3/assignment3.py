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
for i in range(len(reviews)):

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
    
    # list to store the indices of the different dishes
    locs = []
    
    # go through dishes that appear in review
    for dish in dishes:
        
        # start index of search
        begin_index = 0
        
        # loop to locate all occurrences of dish in review
        while begin_index < len(reviews[i]):
            
            # locate first instance of dish
            start_index = reviews[i].find(dish, begin_index)
            
            # if dish not found in review
            if start_index == -1:
                
                break
            
            # add length to get the end index
            end_index = start_index + len(dish)
            
            # store the dish name, start index, end index
            temp = [dish, start_index, end_index]
            
            # boolean to control if we record this dish
            flag = False
            
            # before adding this to list of dishes, check if this dish exists within a dish already found
            for d, start, end in locs:
                
                # check if our current dish starts within another dish
                if start_index >= start and end_index <= end:
                    
                    flag = True
            
            if not flag:
            
                # append to locs
                locs.append(temp)
                
                # update review
                replacement = f"[B-ASP]{dish}[E-ASP]"
                reviews[i] = reviews[i][:start_index] + replacement + reviews[i][end_index:]

            
            # update search index
            begin_index += end_index + len(replacement)
            
            
print(reviews[29])
