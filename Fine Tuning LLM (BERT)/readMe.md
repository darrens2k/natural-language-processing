# Description

This is a small script that I wrote to fine tune BERT using huggingface. The data it accepts is just a small csv file that I had ChatGPT create for me, it classifies statements as whether (1) or not (0) they are axioms. Thus, I am using BERT to perform a binary classification task, hence why I chose a transformer. This code was just to serve as a proof of concept and to play around with huggingface, you'll notice that I only trained for 3 epochs simply because I am just experimenting.

I also have not fact-checked the dataset at all, because once again I am just using this to play around with huggingface.