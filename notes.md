# Plan

## What?

We tell the data story of the changes that occurred in sentiments, tone, 
and overall approach to the public in German newspapers during WWI
(Hamburg mostly, 19th century to early 20th century). Data is presented interactively.

Sentiment analysis & dynamic topic modeling

**remember: Storytelling Starts with Finding the Narrative During the Analysis**

## Tools & inspo
[This](http://www.nytimes.com/newsgraphics/2013/10/13/russia/index.html) kind of visualization, 
with data on he left and text/storytelling on the right as one scrolls.
Tools:
- [graph-scroll](https://github.com/1wheel/graph-scroll)
- field area charts
- streamgraph (D3)

## Task List

- [x] Fetching the data 1914-1918 (if enough). Needs scraping  R
- Cleaning & pre-processing the data. Note: lots of typos  M/R
  - ~~divide by article if possible~~
  - [x] tokenize if needed
  - [ ] get rid of gibberish
- [ ] Computing dynamic topic model R
- [ ] validating its output (Gensim) / importance of war   R/M
- [x] Sentiment analysis on orig data to see sentiments, mood decrease following the quick war excitement  M
- [ ] Story telling text  M
- [ ] On pivotal moments one chart per side, topic/sentiment  R

 ##### Hypothesis
 War importance increases overtime, spirits go down overtime