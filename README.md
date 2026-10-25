# Trust experiment prototype

Prototype for the study "How do order-tracking question difficulty and AI response accuracy affect customer trust among online retail shoppers?" (SE-2431, Zere Boranbay and Kazhmurat Sovet).

It runs one researcher-led session on one laptop: instruction, prior-use question, three scripted chat situations with an order page, a 10-question trust questionnaire (0-40 points), check questions, and a debrief. Design: 2x2 between-subjects (difficulty simple/complex x accuracy correct/incorrect).

## Files
- `index.html` - the whole prototype (plain HTML and JavaScript, no server, no database)
- `scenarios/scenario_pool.csv` - scripted scenarios with word counts (reply length rule: incorrect reply within +-15% of the correct one)

## How to run
Open `index.html` in a browser, or use the GitHub Pages link of this repository. Add `?fast` to the address to shorten the 2 second reply delay for testing only.

## Data
Sessions are saved in the browser of the laptop only. Use "Export CSV" on the setup screen after every session day. No names or contact data are collected. Pilot sessions are flagged in the `pilot` column.

## Scoring
Each of the 10 questions gives 0-4 points by a fixed key (see `TRUST_ITEMS` in `index.html`). Total 0-40: 0-13 distrust, 14-26 mixed, 27-40 trust.
