"""MMAI 5400 - NLP - Assignment 1."""

# MMAI 5400 Assignment 1
# Darren Singh
# 216236275

from bs4 import BeautifulSoup as bSoup
import requests
import csv

# Chosen company is Brita

# grab webpage
webpage = requests.get("https://ca.trustpilot.com/review/brita.co.uk")

# parse through webpage
soup = bSoup(webpage.text, 'html.parser')

# get total amount of reviews, located in a span with a given class name
params = soup.find(name="span", attrs={
                   'class': 'typography_body-l__KUYFJ typography_appea\
                    rance-subtle__8_H2l styles_text__W4hWi'})

# print the contents returned
print("Total number of reviews: ", params.contents[0])

# Get company name from webpage
# locate the span that contains it
name_span = soup.find(name='span', attrs={
                     'class': 'typography_display-s__qOjh6 typography_appearance\
                        -default__AAY17 title_displayName__TtDDM'})
print("Company Name: ", name_span.contents[0])
company_name = name_span.contents[0]

# Total number of reviews
# cast to integer, and remove the comma in the number
tot_reviews = int(params.contents[0].replace(',', ''))

# Code to read reviews from webpage

# counter to track how many reviews have been written
counter = 0

# open a csv file to write to
file = open("reviews.csv", 'w')

# initialize a csv writer object
writer = csv.writer(file, delimiter=",")

# write the header row
writer.writerow(["company_name", "datePublished",
                 "ratingValue", "review_body"])

# while loop to get 600 reviews
while counter <= 600:

    # extract all ratings
    ratings = soup.find_all(
        name="div", attrs={'data-service-review-rating': True})
    # ratings are held within a tag in the div, grab the contents of that tag
    # datetimes of the reviews are also held within the time tag of this div
    rating_vals = []
    review_time = []
    for rating in ratings:
        rating_vals.append(rating['data-service-review-rating'])
        review_time.append(rating.find('time')['datetime'])

    # this code finds the text of all reviews and puts them in a list for now
    review_text = soup.find_all(name='p', attrs={
                               'class': 'typography_body-l__KUYFJ typography_appearance-\
                                default__AAY17 typography_color-black__5LYEn'})
    review_body = []
    for review in review_text:
        review_body.append(review.text)

    # the 3 lists I have assembled contain the required info + \
    # about all reviews on the given page
    # write each element of these lists to the csv file

    # integer to index the lists
    i = 0

    # while loop to iterate through arrays and + \
    # write to csv (each page contains 20 reviews)
    while i < 20:

        # write to csv
        # some reviews contain emoji's, use a try + \
        # and except to skip these reviews
        try:
            # only write to csv if possible
            writer.writerow([str(company_name), str(review_time[i]),
                            str(rating_vals[i]), str(review_body[i])])
        except Exception:
            # continue working through loop if can not write to csv
            i += 1
            continue

        # increment counter
        i += 1

    # url for next page
    page = 'https://ca.trustpilot.com' + \
        soup.find_all(name="a", string='Next page')[0]['href']

    # load next page
    webpage = requests.get(page)

    # parse through next page
    soup = bSoup(webpage.text, 'html.parser')

    # increment loop counter (20 reviews per page)
    counter += 20

# close file
file.close()
