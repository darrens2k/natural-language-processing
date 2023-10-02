# MMAI 5400 Assignment 1
# Darren Singh 
# 216236275

import numpy as np
from bs4 import BeautifulSoup as bs
import requests

### Chosen company is Brita

# company name
companyName = 'Brita'

# grab webpage
webpage = requests.get("https://ca.trustpilot.com/review/brita.co.uk")

# parse through webpage
soup = bs(webpage.text, 'html.parser')

# get total amount of reviews, located in a span with a given class name
params = soup.find(name="span", attrs={'class':'typography_body-l__KUYFJ typography_appearance-subtle__8_H2l styles_text__W4hWi'})

# print the contents returned
print("Total number of reviews: ", params.contents[0])

### Total number of reviews
# cast to integer, and remove the comma in the number
totReviews = int(params.contents[0].replace(',',''))


### Code to read reviews from webpage



# extract all ratings
ratings = soup.find_all(name="div", attrs={'data-service-review-rating':True})
# ratings are held within a tag in the div, grab the contents of that tag
# datetimes of the reviews are also held within the time tag of this div, extract the times
ratingVals = []
reviewTime = []
for rating in ratings:
    ratingVals.append(rating['data-service-review-rating'])
    reviewTime.append(rating.find('time')['datetime'])

# this code finds the text of all reviews and puts them in a list for now
reviewText = soup.find_all(name='p', attrs={'class':'typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn'})
reviewBody = []
for review in reviewText:
    reviewBody.append(review.text)

# for a given review, extract the rating out of 5

# page = 'https://ca.trustpilot.com' + soup.find_all(name="a", string='Next page')[0]['href']
# print(page)